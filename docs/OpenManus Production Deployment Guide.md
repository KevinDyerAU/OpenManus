# OpenManus Production Deployment Guide

## Overview

This comprehensive guide covers the production deployment of OpenManus across multiple platforms with enterprise-grade configurations, monitoring, and security features. The deployment supports local Docker environments, Render platform, and AWS cloud infrastructure with full CI/CD automation.

## Architecture Overview

The production deployment architecture consists of several key components designed for scalability, reliability, and maintainability:

### Core Services
- **FastAPI Application**: Main OpenManus API server with WebSocket support
- **React Frontend**: Modern web interface built with Vite and served by the API
- **PostgreSQL Database**: Primary data storage with automated backups
- **Redis Cache**: Session storage, caching, and Celery message broker
- **Nginx Reverse Proxy**: Load balancing, SSL termination, and static file serving
- **Celery Workers**: Background task processing and workflow execution

### Monitoring Stack
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization dashboards and monitoring
- **ELK Stack**: Centralized logging with Elasticsearch, Logstash, and Kibana
- **Health Checks**: Automated health monitoring and recovery

### Security Features
- **SSL/TLS Encryption**: End-to-end encryption with automatic certificate management
- **JWT Authentication**: Secure token-based authentication with role-based access
- **Rate Limiting**: API and WebSocket rate limiting to prevent abuse
- **Security Headers**: Comprehensive security headers and CORS protection
- **Input Validation**: Strict input validation and sanitization

## Deployment Platforms

### 1. Local Docker Deployment

The local Docker deployment provides a complete production-like environment for development and testing.

#### Prerequisites
- Docker 24.0+ with Docker Compose
- 8GB+ RAM and 20GB+ disk space
- Git for version control

#### Quick Start

```bash
# Clone the repository
git clone https://github.com/FoundationAgents/OpenManus.git
cd OpenManus

# Run the deployment script
chmod +x scripts/deployment/deploy-production.sh
./scripts/deployment/deploy-production.sh local production
```

#### Manual Setup

```bash
# Create environment file
cp .env.example .env.production

# Edit configuration
nano .env.production

# Set required variables
export POSTGRES_PASSWORD="your-secure-password"
export OPENROUTER_API_KEY="your-openrouter-key"
export JWT_SECRET="your-jwt-secret-32-chars-min"

# Build and start services
docker-compose -f docker-compose.prod.yml up -d

# Check service status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f openmanus
```

#### Service URLs
- **Application**: https://localhost:8000
- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **Kibana Logs**: http://localhost:5601
- **Prometheus Metrics**: http://localhost:9091

#### Configuration Files
- `docker-compose.prod.yml`: Main service orchestration
- `.env.production`: Environment variables
- `config/nginx/nginx.conf`: Nginx configuration
- `config/prometheus/prometheus.yml`: Monitoring configuration

### 2. Render Platform Deployment

Render provides a managed platform with automatic scaling, SSL certificates, and integrated monitoring.

#### Cost Estimation
- **Web Service (Standard)**: $25/month × 2 instances = $50/month
- **Worker Service (Starter)**: $7/month
- **Scheduler Service (Starter)**: $7/month
- **PostgreSQL (Standard)**: $20/month
- **Redis (Standard)**: $30/month
- **Total**: ~$114/month

#### Prerequisites
- Render account with payment method
- GitHub repository with OpenManus code
- Domain name (optional)

#### Deployment Steps

1. **Prepare Repository**
```bash
# Ensure render-production.yaml is in repository root
git add render-production.yaml
git commit -m "Add Render production configuration"
git push origin main
```

2. **Deploy via Render Dashboard**
- Connect GitHub repository to Render
- Import services from `render-production.yaml`
- Set environment variables in Render dashboard:
  - `OPENROUTER_API_KEY`
  - `JWT_SECRET`
  - `SENTRY_DSN` (optional)

3. **Deploy via CLI**
```bash
# Install Render CLI
npm install -g @render/cli

# Authenticate
render auth login

# Deploy services
render deploy --file render-production.yaml
```

4. **Configure Custom Domain** (Optional)
- Add domain in Render dashboard
- Update DNS records:
  - `A` record: `@` → Render IP
  - `CNAME` record: `www` → `your-app.onrender.com`

#### Monitoring and Scaling
- **Auto-scaling**: Configured for 2-10 instances based on CPU usage
- **Health Checks**: Automatic health monitoring with `/health` endpoint
- **Logs**: Centralized logging in Render dashboard
- **Metrics**: Built-in performance metrics and alerts

### 3. AWS Cloud Deployment

AWS deployment provides enterprise-grade infrastructure with full control over scaling, security, and monitoring.

#### Cost Estimation (Monthly)
- **ECS Fargate**: $50-100 (2-4 tasks)
- **RDS PostgreSQL**: $40-80 (db.t3.medium)
- **ElastiCache Redis**: $30-60 (cache.t3.medium)
- **Application Load Balancer**: $20
- **NAT Gateway**: $45 (2 AZs)
- **Data Transfer**: $10-20
- **CloudWatch**: $10-20
- **Total**: ~$205-345/month

#### Prerequisites
- AWS account with appropriate permissions
- AWS CLI configured with credentials
- Docker for building images
- Domain name and ACM certificate (optional)

#### Deployment Steps

1. **Set Environment Variables**
```bash
export AWS_REGION="us-east-1"
export POSTGRES_PASSWORD="your-secure-password"
export OPENROUTER_API_KEY="your-openrouter-key"
export JWT_SECRET="your-jwt-secret-32-chars-min"
export DOMAIN_NAME="your-domain.com"  # Optional
export CERTIFICATE_ARN="arn:aws:acm:..."  # Optional
```

2. **Deploy Infrastructure**
```bash
# Deploy using the automation script
./scripts/deployment/deploy-production.sh aws production

# Or deploy manually with CloudFormation
aws cloudformation deploy \
  --template-file aws-production-cloudformation.yaml \
  --stack-name openmanus-production \
  --parameter-overrides \
    Environment=production \
    DBPassword=$POSTGRES_PASSWORD \
    OpenRouterApiKey=$OPENROUTER_API_KEY \
    JWTSecret=$JWT_SECRET \
    DomainName=$DOMAIN_NAME \
    CertificateArn=$CERTIFICATE_ARN \
  --capabilities CAPABILITY_IAM \
  --region $AWS_REGION
```

3. **Configure Domain** (Optional)
```bash
# Get Load Balancer DNS name
ALB_DNS=$(aws cloudformation describe-stacks \
  --stack-name openmanus-production \
  --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerDNS`].OutputValue' \
  --output text)

# Update DNS records
# A record: your-domain.com → ALB_DNS
# CNAME record: www.your-domain.com → ALB_DNS
```

#### AWS Services Used
- **ECS Fargate**: Serverless container orchestration
- **RDS PostgreSQL**: Managed database with Multi-AZ
- **ElastiCache Redis**: Managed Redis cluster
- **Application Load Balancer**: Traffic distribution and SSL termination
- **VPC**: Isolated network with public/private subnets
- **CloudWatch**: Monitoring, logging, and alerting
- **Secrets Manager**: Secure secret storage
- **S3**: File storage and backups

## CI/CD Pipeline

The GitHub Actions pipeline provides automated testing, building, and deployment with comprehensive quality checks.

### Pipeline Stages

1. **Code Quality and Testing**
   - Python linting with flake8
   - Type checking with mypy
   - Security scanning with bandit
   - Frontend linting and type checking
   - Unit and integration tests
   - Coverage reporting

2. **Security Scanning**
   - Trivy vulnerability scanning
   - CodeQL static analysis
   - Dependency vulnerability checks
   - Container image scanning

3. **Build and Push**
   - Multi-architecture Docker builds
   - Container registry push
   - Image vulnerability scanning
   - Artifact management

4. **Deployment**
   - Automated deployment to Render
   - Manual deployment to AWS
   - Health checks and validation
   - Rollback on failure

5. **Post-Deployment**
   - Integration testing
   - Load testing
   - Monitoring setup
   - Notification and alerting

### Required Secrets

Configure these secrets in your GitHub repository:

#### General Secrets
```bash
OPENROUTER_API_KEY=your_openrouter_api_key
JWT_SECRET=your_jwt_secret_32_chars_minimum
POSTGRES_PASSWORD=your_secure_database_password
```

#### Render Deployment
```bash
RENDER_API_KEY=your_render_api_key
```

#### AWS Deployment
```bash
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
DOMAIN_NAME=your-domain.com  # Optional
CERTIFICATE_ARN=arn:aws:acm:...  # Optional
```

#### Monitoring (Optional)
```bash
SENTRY_DSN=your_sentry_dsn
DATADOG_API_KEY=your_datadog_api_key
SLACK_WEBHOOK_URL=your_slack_webhook
```

### Triggering Deployments

```bash
# Automatic deployment on main branch push
git push origin main

# Manual deployment via GitHub Actions
# Go to Actions tab → Production Deployment → Run workflow

# Tag-based deployment
git tag v1.0.0
git push origin v1.0.0
```

## Environment Configuration

### Required Environment Variables

#### Application Configuration
```bash
APP_NAME=OpenManus
APP_VERSION=1.0.0
APP_ENVIRONMENT=production
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
```

#### Database Configuration
```bash
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30
```

#### Authentication Configuration
```bash
JWT_SECRET=your-jwt-secret-32-characters-minimum
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
```

#### OpenRouter Configuration
```bash
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_SITE_URL=https://your-site.com
OPENROUTER_SITE_NAME=Your App Name
```

#### Security Configuration
```bash
CORS_ORIGINS=https://your-domain.com,https://www.your-domain.com
RATE_LIMIT=100
MAX_REQUEST_SIZE=10485760
```

### Optional Configuration

#### Monitoring
```bash
SENTRY_DSN=your_sentry_dsn
ENABLE_METRICS=true
LOG_LEVEL=INFO
```

#### Features
```bash
ENABLE_BROWSER_TOOLS=true
ENABLE_FLOW_EXECUTION=true
ENABLE_MCP_TOOLS=true
ENABLE_STREAMING=true
```

## Monitoring and Observability

### Metrics Collection

The deployment includes comprehensive metrics collection using Prometheus:

#### Application Metrics
- Request rate, latency, and error rate
- WebSocket connection count and duration
- Database connection pool usage
- Cache hit/miss ratios
- Background task queue length

#### Infrastructure Metrics
- CPU, memory, and disk usage
- Network traffic and latency
- Container resource utilization
- Database performance metrics
- Load balancer health and traffic

#### Custom Metrics
```python
# Example custom metrics in application
from prometheus_client import Counter, Histogram, Gauge

request_count = Counter('openmanus_requests_total', 'Total requests', ['method', 'endpoint'])
request_duration = Histogram('openmanus_request_duration_seconds', 'Request duration')
active_connections = Gauge('openmanus_websocket_connections', 'Active WebSocket connections')
```

### Logging Strategy

Centralized logging with structured JSON format:

#### Log Levels
- **ERROR**: Application errors and exceptions
- **WARN**: Performance issues and deprecation warnings
- **INFO**: Request logs and business events
- **DEBUG**: Detailed debugging information (development only)

#### Log Aggregation
- **Local**: Docker logs with log rotation
- **Render**: Built-in log aggregation
- **AWS**: CloudWatch Logs with retention policies

#### Log Analysis
- **Kibana**: Log search and visualization
- **Grafana**: Log-based alerting and dashboards
- **CloudWatch Insights**: AWS log analysis

### Alerting Rules

#### Critical Alerts
- Application down or unhealthy
- Database connection failures
- High error rate (>5% for 5 minutes)
- Memory usage >90% for 10 minutes
- Disk usage >85%

#### Warning Alerts
- High response time (>2s for 10 minutes)
- High CPU usage (>80% for 15 minutes)
- Database slow queries
- Cache miss rate >50%

#### Alert Channels
- **Email**: Critical alerts to operations team
- **Slack**: All alerts to development channel
- **PagerDuty**: Critical alerts for on-call rotation
- **SMS**: Critical production outages

## Security Considerations

### Network Security
- **VPC Isolation**: Private subnets for application and database
- **Security Groups**: Restrictive firewall rules
- **WAF**: Web Application Firewall for DDoS protection
- **SSL/TLS**: End-to-end encryption with strong ciphers

### Application Security
- **Authentication**: JWT tokens with secure generation
- **Authorization**: Role-based access control
- **Input Validation**: Strict validation and sanitization
- **Rate Limiting**: API and WebSocket rate limiting
- **CORS**: Restrictive cross-origin policies

### Data Security
- **Encryption at Rest**: Database and file storage encryption
- **Encryption in Transit**: TLS for all communications
- **Secret Management**: Secure storage of API keys and passwords
- **Backup Encryption**: Encrypted database backups

### Compliance
- **GDPR**: Data protection and privacy controls
- **SOC 2**: Security and availability controls
- **HIPAA**: Healthcare data protection (if applicable)
- **PCI DSS**: Payment data security (if applicable)

## Backup and Disaster Recovery

### Backup Strategy

#### Database Backups
- **Frequency**: Daily automated backups
- **Retention**: 30 days for daily, 12 months for monthly
- **Storage**: Encrypted S3 storage with cross-region replication
- **Testing**: Monthly backup restoration tests

#### Application Data
- **File Uploads**: Synchronized to S3 with versioning
- **Configuration**: Version controlled in Git
- **Secrets**: Backed up in secure secret management

#### Backup Commands
```bash
# Manual database backup
docker-compose exec postgres pg_dump -U openmanus openmanus > backup.sql

# Restore from backup
docker-compose exec postgres psql -U openmanus openmanus < backup.sql

# S3 backup sync
aws s3 sync /app/data s3://openmanus-backups/data/
```

### Disaster Recovery

#### Recovery Time Objectives (RTO)
- **Local**: 15 minutes (container restart)
- **Render**: 5 minutes (automatic failover)
- **AWS**: 10 minutes (multi-AZ deployment)

#### Recovery Point Objectives (RPO)
- **Database**: 1 hour (continuous replication)
- **Files**: 24 hours (daily sync)
- **Configuration**: 0 (version controlled)

#### Disaster Recovery Procedures
1. **Assess Impact**: Determine scope and severity
2. **Activate DR Plan**: Switch to backup systems
3. **Restore Data**: Restore from latest backups
4. **Validate System**: Run health checks and tests
5. **Update DNS**: Point traffic to recovery environment
6. **Monitor**: Continuous monitoring during recovery

## Performance Optimization

### Application Performance
- **Connection Pooling**: Database and Redis connection pools
- **Caching**: Multi-level caching strategy
- **Async Processing**: Non-blocking I/O and background tasks
- **Code Optimization**: Profiling and performance tuning

### Infrastructure Performance
- **Auto-scaling**: Horizontal scaling based on metrics
- **Load Balancing**: Traffic distribution across instances
- **CDN**: Content delivery network for static assets
- **Database Optimization**: Query optimization and indexing

### Monitoring Performance
- **Response Time**: P95 response time <500ms
- **Throughput**: 1000+ requests per second
- **Availability**: 99.9% uptime SLA
- **Error Rate**: <0.1% error rate

## Troubleshooting Guide

### Common Issues

#### Application Won't Start
```bash
# Check container logs
docker-compose logs openmanus

# Check environment variables
docker-compose exec openmanus env | grep -E "(DATABASE|REDIS|OPENROUTER)"

# Verify database connection
docker-compose exec openmanus python -c "from app.database import engine; print(engine.url)"
```

#### Database Connection Issues
```bash
# Check PostgreSQL status
docker-compose exec postgres pg_isready -U openmanus

# Check connection from application
docker-compose exec openmanus python -c "
import asyncpg
import asyncio
async def test():
    conn = await asyncpg.connect('postgresql://openmanus:password@postgres:5432/openmanus')
    print('Connected successfully')
    await conn.close()
asyncio.run(test())
"
```

#### High Memory Usage
```bash
# Check container memory usage
docker stats

# Check application memory usage
docker-compose exec openmanus python -c "
import psutil
print(f'Memory usage: {psutil.virtual_memory().percent}%')
"

# Restart services if needed
docker-compose restart openmanus
```

#### WebSocket Connection Issues
```bash
# Test WebSocket connection
wscat -c ws://localhost:8000/ws/chat

# Check Nginx WebSocket configuration
docker-compose exec nginx nginx -t

# Check application WebSocket handling
docker-compose logs openmanus | grep -i websocket
```

### Performance Issues

#### Slow Response Times
1. Check database query performance
2. Verify cache hit rates
3. Monitor CPU and memory usage
4. Check network latency
5. Review application logs for bottlenecks

#### High Error Rates
1. Check application error logs
2. Verify external service availability
3. Monitor rate limiting
4. Check database connection pool
5. Review recent deployments

### Recovery Procedures

#### Service Recovery
```bash
# Restart all services
docker-compose restart

# Rebuild and restart specific service
docker-compose up -d --build openmanus

# Check service health
curl -f http://localhost:8000/health
```

#### Database Recovery
```bash
# Restore from backup
docker-compose exec postgres psql -U openmanus openmanus < backup.sql

# Rebuild database from migrations
docker-compose exec openmanus alembic upgrade head
```

## Maintenance Procedures

### Regular Maintenance

#### Daily Tasks
- Monitor system health and alerts
- Review error logs and metrics
- Check backup completion
- Verify security updates

#### Weekly Tasks
- Review performance metrics
- Update dependencies
- Clean up old logs and files
- Test disaster recovery procedures

#### Monthly Tasks
- Security vulnerability assessment
- Performance optimization review
- Backup restoration testing
- Documentation updates

### Update Procedures

#### Application Updates
```bash
# Pull latest code
git pull origin main

# Build new image
docker-compose build openmanus

# Deploy with zero downtime
docker-compose up -d --no-deps openmanus
```

#### Security Updates
```bash
# Update base images
docker-compose pull

# Rebuild with latest security patches
docker-compose build --no-cache

# Deploy updates
docker-compose up -d
```

#### Database Migrations
```bash
# Run migrations
docker-compose exec openmanus alembic upgrade head

# Verify migration success
docker-compose exec openmanus alembic current
```

This comprehensive production deployment guide provides everything needed to deploy, monitor, and maintain OpenManus in production environments across multiple platforms with enterprise-grade reliability, security, and performance.

