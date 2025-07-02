"""
Docker configuration for GNN Trading System
Production-ready containerized deployment
"""

# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements-dev.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY setup.py .
COPY README.md .

# Install the package
RUN pip install -e .

# Create non-root user for security
RUN groupadd -r gnntrading && \
    useradd -r -g gnntrading -d /app -s /bin/bash gnntrading && \
    chown -R gnntrading:gnntrading /app

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/models && \
    chown -R gnntrading:gnntrading /app/data /app/logs /app/models

# Switch to non-root user
USER gnntrading

# Expose port for API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden)
CMD ["uvicorn", "gnn_trading.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
