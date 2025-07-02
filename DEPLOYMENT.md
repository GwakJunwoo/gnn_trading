# GNN Trading System - Production Deployment Guide

## Overview

This guide covers the complete deployment of the GNN Trading System in production environments. The system is designed for high-availability, scalability, and commercial deployment.

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended), Windows Server 2019+, or macOS 12+
- **CPU**: 8+ cores recommended for production
- **Memory**: 16GB+ RAM (32GB+ recommended with large models)
- **GPU**: Optional but recommended for model training (NVIDIA with CUDA 11.8+)
- **Storage**: 100GB+ SSD for data and models
- **Network**: High-speed internet connection for real-time data feeds

### Software Dependencies
- Docker 20.10+
- Docker Compose 2.0+
- Python 3.11+ (if installing locally)
- PostgreSQL 15+ (or use Docker)
- Redis 7+ (or use Docker)

## Deployment Options

### Option 1: Docker Compose (Recommended for Small-Medium Scale)

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd gnn_trading
   ```

2. **Environment Configuration**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit environment variables
   nano .env
   ```

3. **Configure Secrets**
   ```bash
   # Generate secure secret key
   export SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
   
   # Set database password
   export POSTGRES_PASSWORD=$(python -c "import secrets; print(secrets.token_urlsafe(16))")
   ```

4. **Deploy Services**
   ```bash
   # Build and start all services
   ./scripts/deploy.sh deploy-local
   
   # Check service health
   ./scripts/deploy.sh status
   ```

5. **Verify Deployment**
   ```bash
   # Test API health
   curl http://localhost:8000/health
   
   # Access monitoring
   # Grafana: http://localhost:3000 (admin/admin123)
   # Prometheus: http://localhost:9090
   ```

### Option 2: Kubernetes (Recommended for Large Scale)

1. **Prerequisites**
   - Kubernetes cluster (1.24+)
   - Helm 3.0+
   - kubectl configured

2. **Deploy with Helm**
   ```bash
   # Add helm chart
   helm repo add gnn-trading ./helm/gnn-trading
   
   # Install with production values
   helm install gnn-trading gnn-trading/gnn-trading \
     --namespace gnn-trading \
     --create-namespace \
     --values helm/values-production.yaml
   ```

3. **Configure Ingress**
   ```bash
   # Apply ingress configuration
   kubectl apply -f k8s/ingress.yaml
   ```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ENV` | Environment (development/staging/production) | development | Yes |
| `SECRET_KEY` | Application secret key | - | Yes (Production) |
| `POSTGRES_HOST` | PostgreSQL host | postgres | No |
| `POSTGRES_PASSWORD` | PostgreSQL password | - | Yes |
| `REDIS_HOST` | Redis host | redis | No |
| `API_WORKERS` | Number of API workers | 4 | No |
| `MONITORING_ENABLED` | Enable monitoring | true | No |

### Configuration Files

#### Production Configuration (`configs/production_config.yaml`)
- Database connection pooling
- Redis caching configuration
- API rate limiting
- Security settings
- Monitoring and alerting

#### Security Configuration
- JWT token configuration
- API key authentication
- CORS settings
- HTTPS enforcement
- Rate limiting rules

## Security

### Authentication & Authorization
```python
# API Key Authentication
headers = {
    'X-API-Key': 'your-api-key',
    'Content-Type': 'application/json'
}
```

### JWT Tokens
```python
# JWT Token Authentication
headers = {
    'Authorization': 'Bearer your-jwt-token',
    'Content-Type': 'application/json'
}
```

### Network Security
- Use HTTPS in production
- Configure firewall rules
- VPN access for administrative interfaces
- Network segmentation

## Monitoring & Alerting

### Prometheus Metrics
- API response times and error rates
- Model prediction accuracy
- System resource usage
- Data quality scores
- Business metrics

### Grafana Dashboards
- System overview dashboard
- Model performance dashboard
- Business intelligence dashboard
- Alert management dashboard

### Alert Configuration
```yaml
# Example alert rule
alert_rules:
  - name: "High API Latency"
    condition: "api_latency_p95 > 1000"
    threshold: 1000
    severity: "warning"
    description: "API response time is too high"
```

### Notification Channels
- Email alerts
- Slack notifications
- Webhook integrations
- SMS alerts (via third-party)

## Backup & Recovery

### Automated Backups
```bash
# Daily database backup
./scripts/deploy.sh backup

# Backup models and configurations
tar -czf models-backup-$(date +%Y%m%d).tar.gz models/ configs/
```

### Recovery Procedures
```bash
# Restore from backup
./scripts/deploy.sh restore backup-20231201.tar.gz

# Restore database
docker exec -i postgres psql -U gnn_user gnn_trading < backup/database.sql
```

## Scaling

### Horizontal Scaling
- Multiple API instances behind load balancer
- Separate worker processes for background tasks
- Redis cluster for distributed caching
- Database read replicas

### Vertical Scaling
- Increase CPU/memory allocation
- GPU acceleration for model inference
- SSD storage for better I/O performance

### Auto-scaling Configuration
```yaml
# Kubernetes HPA example
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gnn-trading-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gnn-trading-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Performance Optimization

### Database Optimization
- Connection pooling
- Query optimization
- Index management
- Partitioning for large datasets

### Caching Strategy
- Redis for session data
- Application-level caching
- CDN for static assets
- Model result caching

### Model Optimization
- Model quantization
- Batch inference
- GPU acceleration
- Model ensembling

## Troubleshooting

### Common Issues

#### API Not Responding
```bash
# Check container status
docker-compose ps

# Check logs
docker-compose logs gnn-trading-api

# Restart service
docker-compose restart gnn-trading-api
```

#### Database Connection Issues
```bash
# Check database status
docker-compose exec postgres pg_isready

# Check connection
docker-compose exec gnn-trading-api python -c "
from gnn_trading.config import get_config
print(get_config().get_database_url())
"
```

#### Memory Issues
```bash
# Check memory usage
docker stats

# Clear cache
docker-compose exec redis redis-cli FLUSHALL

# Restart with more memory
docker-compose up -d --scale gnn-trading-api=1
```

### Log Analysis
```bash
# View API logs
docker-compose logs -f gnn-trading-api

# View system logs
docker-compose logs -f

# Search for errors
docker-compose logs | grep ERROR
```

## Maintenance

### Regular Tasks
- Daily backup verification
- Weekly security updates
- Monthly performance review
- Quarterly disaster recovery testing

### Update Procedures
```bash
# Update to new version
git pull origin main
docker-compose build
docker-compose up -d

# Rollback if needed
docker-compose down
docker-compose up -d --scale gnn-trading-api=1
```

### Health Checks
```bash
# API health check
curl -f http://localhost:8000/health

# Database health check
docker-compose exec postgres pg_isready

# Redis health check
docker-compose exec redis redis-cli ping
```

## Support & Documentation

### API Documentation
- Swagger UI: `http://your-domain/docs`
- ReDoc: `http://your-domain/redoc`
- OpenAPI spec: `http://your-domain/openapi.json`

### Monitoring URLs
- Grafana: `http://your-domain:3000`
- Prometheus: `http://your-domain:9090`
- API Metrics: `http://your-domain:8000/metrics`

### Support Contacts
- Technical Support: support@gnn-trading.com
- Emergency Hotline: +1-XXX-XXX-XXXX
- Documentation: https://docs.gnn-trading.com

## License & Commercial Use

This GNN Trading System is designed for commercial deployment. Please ensure compliance with:

- Software licensing terms
- Data usage agreements
- Financial regulations (if applicable)
- Privacy requirements (GDPR, etc.)

For commercial licensing and support packages, contact: sales@gnn-trading.com

---

**Note**: This deployment guide assumes a production-ready environment. Always test thoroughly in staging before deploying to production.
