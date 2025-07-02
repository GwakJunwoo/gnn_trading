"""
Deployment and infrastructure management scripts
"""

#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="gnn-trading"
DOCKER_REGISTRY="your-registry.com"
VERSION=${1:-latest}

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_info "All dependencies are satisfied"
}

build_image() {
    log_info "Building Docker image..."
    
    # Build the main image
    docker build -t ${PROJECT_NAME}:${VERSION} .
    
    # Tag for registry if specified
    if [ ! -z "$DOCKER_REGISTRY" ]; then
        docker tag ${PROJECT_NAME}:${VERSION} ${DOCKER_REGISTRY}/${PROJECT_NAME}:${VERSION}
        docker tag ${PROJECT_NAME}:${VERSION} ${DOCKER_REGISTRY}/${PROJECT_NAME}:latest
    fi
    
    log_info "Image built successfully"
}

push_image() {
    if [ -z "$DOCKER_REGISTRY" ]; then
        log_warn "No Docker registry specified, skipping push"
        return
    fi
    
    log_info "Pushing image to registry..."
    
    docker push ${DOCKER_REGISTRY}/${PROJECT_NAME}:${VERSION}
    docker push ${DOCKER_REGISTRY}/${PROJECT_NAME}:latest
    
    log_info "Image pushed successfully"
}

deploy_local() {
    log_info "Deploying locally with Docker Compose..."
    
    # Create necessary directories
    mkdir -p data logs models monitoring/grafana/dashboards
    
    # Start services
    docker-compose up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Check health
    if curl -f http://localhost:8000/health &> /dev/null; then
        log_info "API is healthy"
    else
        log_error "API health check failed"
        docker-compose logs gnn-trading-api
        exit 1
    fi
    
    log_info "Local deployment successful"
    log_info "API: http://localhost:8000"
    log_info "Grafana: http://localhost:3000 (admin/admin123)"
    log_info "Prometheus: http://localhost:9090"
}

deploy_production() {
    log_info "Deploying to production..."
    
    # This would typically involve:
    # 1. Kubernetes deployment
    # 2. Helm charts
    # 3. Infrastructure as Code (Terraform)
    # 4. CI/CD pipeline integration
    
    log_warn "Production deployment not implemented in this script"
    log_info "Please use your production deployment pipeline"
}

run_tests() {
    log_info "Running tests..."
    
    # Run tests in container
    docker run --rm -v $(pwd):/app ${PROJECT_NAME}:${VERSION} python -m pytest tests/ -v --cov=src/gnn_trading --cov-report=html
    
    log_info "Tests completed"
}

cleanup() {
    log_info "Cleaning up..."
    
    # Stop and remove containers
    docker-compose down -v
    
    # Remove unused images
    docker image prune -f
    
    log_info "Cleanup completed"
}

backup_data() {
    log_info "Backing up data..."
    
    # Create backup directory
    BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p $BACKUP_DIR
    
    # Backup PostgreSQL data
    docker exec gnn-trading-postgres pg_dump -U gnn_user gnn_trading > $BACKUP_DIR/database.sql
    
    # Backup models and configs
    cp -r models $BACKUP_DIR/
    cp -r configs $BACKUP_DIR/
    
    # Compress backup
    tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR
    rm -rf $BACKUP_DIR
    
    log_info "Backup completed: $BACKUP_DIR.tar.gz"
}

restore_data() {
    BACKUP_FILE=$1
    
    if [ -z "$BACKUP_FILE" ]; then
        log_error "Please specify backup file"
        exit 1
    fi
    
    log_info "Restoring data from $BACKUP_FILE..."
    
    # Extract backup
    TEMP_DIR=$(mktemp -d)
    tar -xzf $BACKUP_FILE -C $TEMP_DIR
    
    # Restore database
    docker exec -i gnn-trading-postgres psql -U gnn_user gnn_trading < $TEMP_DIR/database.sql
    
    # Restore models and configs
    cp -r $TEMP_DIR/models ./
    cp -r $TEMP_DIR/configs ./
    
    # Cleanup
    rm -rf $TEMP_DIR
    
    log_info "Data restored successfully"
}

show_status() {
    log_info "Service status:"
    docker-compose ps
    
    log_info "Resource usage:"
    docker stats --no-stream
}

show_logs() {
    SERVICE=${1:-gnn-trading-api}
    LINES=${2:-100}
    
    log_info "Showing logs for $SERVICE (last $LINES lines):"
    docker-compose logs --tail=$LINES $SERVICE
}

# Main script
case "$1" in
    "build")
        check_dependencies
        build_image
        ;;
    "deploy-local")
        check_dependencies
        build_image
        deploy_local
        ;;
    "deploy-prod")
        check_dependencies
        build_image
        push_image
        deploy_production
        ;;
    "test")
        check_dependencies
        build_image
        run_tests
        ;;
    "cleanup")
        cleanup
        ;;
    "backup")
        backup_data
        ;;
    "restore")
        restore_data $2
        ;;
    "status")
        show_status
        ;;
    "logs")
        show_logs $2 $3
        ;;
    *)
        echo "Usage: $0 {build|deploy-local|deploy-prod|test|cleanup|backup|restore|status|logs}"
        echo ""
        echo "Commands:"
        echo "  build         - Build Docker image"
        echo "  deploy-local  - Deploy locally with Docker Compose"
        echo "  deploy-prod   - Deploy to production"
        echo "  test          - Run tests"
        echo "  cleanup       - Clean up containers and images"
        echo "  backup        - Backup data"
        echo "  restore FILE  - Restore data from backup"
        echo "  status        - Show service status"
        echo "  logs [SERVICE] [LINES] - Show service logs"
        exit 1
        ;;
esac
