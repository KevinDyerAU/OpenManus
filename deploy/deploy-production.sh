#!/bin/bash

# OpenManus Production Deployment Script
# Supports local Docker, Render, and AWS deployments

set -euo pipefail

# ================================
# Configuration
# ================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOYMENT_TYPE="${1:-local}"
ENVIRONMENT="${2:-production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ================================
# Utility Functions
# ================================
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        error "Command '$1' is required but not installed."
    fi
}

check_file() {
    if [[ ! -f "$1" ]]; then
        error "Required file '$1' not found."
    fi
}

check_env_var() {
    if [[ -z "${!1:-}" ]]; then
        error "Environment variable '$1' is required but not set."
    fi
}

# ================================
# Pre-deployment Checks
# ================================
pre_deployment_checks() {
    log "Running pre-deployment checks..."
    
    # Check required commands
    check_command "docker"
    check_command "git"
    
    # Check project structure
    check_file "$PROJECT_ROOT/Dockerfile.production"
    check_file "$PROJECT_ROOT/docker-compose.prod.yml"
    check_file "$PROJECT_ROOT/requirements-prod.txt"
    
    # Check environment file
    if [[ "$DEPLOYMENT_TYPE" == "local" ]]; then
        check_file "$PROJECT_ROOT/.env.production"
    fi
    
    # Check Git status
    if [[ -n "$(git status --porcelain)" ]]; then
        warning "Working directory has uncommitted changes."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            error "Deployment cancelled."
        fi
    fi
    
    success "Pre-deployment checks passed."
}

# ================================
# Build Application
# ================================
build_application() {
    log "Building application..."
    
    cd "$PROJECT_ROOT"
    
    # Build frontend
    log "Building frontend..."
    cd openmanus-chat
    if [[ ! -f "package.json" ]]; then
        error "Frontend package.json not found."
    fi
    
    # Install dependencies
    if command -v pnpm &> /dev/null; then
        pnpm install --frozen-lockfile
        pnpm run build
    elif command -v npm &> /dev/null; then
        npm ci
        npm run build
    else
        error "Neither pnpm nor npm found."
    fi
    
    cd "$PROJECT_ROOT"
    
    # Build Docker image
    log "Building Docker image..."
    docker build -f Dockerfile.production -t openmanus:$ENVIRONMENT .
    
    success "Application built successfully."
}

# ================================
# Local Docker Deployment
# ================================
deploy_local() {
    log "Deploying to local Docker environment..."
    
    # Check required environment variables
    if [[ ! -f ".env.production" ]]; then
        warning ".env.production not found. Creating from template..."
        cp .env.example .env.production
        warning "Please edit .env.production with your configuration."
    fi
    
    # Create necessary directories
    mkdir -p logs uploads downloads data backups certs
    
    # Generate SSL certificates if not present
    if [[ ! -f "certs/cert.pem" ]]; then
        log "Generating self-signed SSL certificates..."
        mkdir -p certs
        openssl req -x509 -newkey rsa:4096 -keyout certs/key.pem -out certs/cert.pem -days 365 -nodes \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
    fi
    
    # Start services
    log "Starting services with Docker Compose..."
    docker-compose -f docker-compose.prod.yml up -d
    
    # Wait for services to be ready
    log "Waiting for services to be ready..."
    sleep 30
    
    # Health check
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            success "Application is healthy and ready!"
            break
        fi
        
        log "Attempt $attempt/$max_attempts: Waiting for application to be ready..."
        sleep 10
        ((attempt++))
    done
    
    if [[ $attempt -gt $max_attempts ]]; then
        error "Application failed to become ready within timeout."
    fi
    
    # Display deployment information
    echo
    success "Local deployment completed successfully!"
    echo "Application URL: http://localhost:8000"
    echo "Grafana Dashboard: http://localhost:3000"
    echo "Kibana Logs: http://localhost:5601"
    echo
    echo "To view logs: docker-compose -f docker-compose.prod.yml logs -f"
    echo "To stop: docker-compose -f docker-compose.prod.yml down"
}

# ================================
# Render Deployment
# ================================
deploy_render() {
    log "Deploying to Render..."
    
    # Check Render CLI
    if ! command -v render &> /dev/null; then
        log "Installing Render CLI..."
        npm install -g @render/cli
    fi
    
    # Check authentication
    if ! render auth whoami &> /dev/null; then
        error "Please authenticate with Render CLI: render auth login"
    fi
    
    # Deploy using render.yaml
    if [[ -f "render-production.yaml" ]]; then
        log "Deploying with render-production.yaml..."
        render deploy --file render-production.yaml
    else
        error "render-production.yaml not found."
    fi
    
    success "Render deployment initiated. Check Render dashboard for status."
}

# ================================
# AWS Deployment
# ================================
deploy_aws() {
    log "Deploying to AWS..."
    
    # Check AWS CLI
    check_command "aws"
    
    # Check AWS authentication
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS CLI not configured. Run 'aws configure' first."
    fi
    
    # Set AWS region
    local aws_region="${AWS_REGION:-us-east-1}"
    local stack_name="openmanus-$ENVIRONMENT"
    
    # Check required environment variables
    check_env_var "POSTGRES_PASSWORD"
    check_env_var "OPENROUTER_API_KEY"
    check_env_var "JWT_SECRET"
    
    # Build and push Docker image to ECR
    log "Building and pushing Docker image to ECR..."
    
    # Create ECR repository if it doesn't exist
    local ecr_repo="openmanus"
    local account_id=$(aws sts get-caller-identity --query Account --output text)
    local ecr_uri="$account_id.dkr.ecr.$aws_region.amazonaws.com/$ecr_repo"
    
    aws ecr describe-repositories --repository-names $ecr_repo --region $aws_region &> /dev/null || \
        aws ecr create-repository --repository-name $ecr_repo --region $aws_region
    
    # Login to ECR
    aws ecr get-login-password --region $aws_region | docker login --username AWS --password-stdin $ecr_uri
    
    # Tag and push image
    docker tag openmanus:$ENVIRONMENT $ecr_uri:$ENVIRONMENT
    docker tag openmanus:$ENVIRONMENT $ecr_uri:latest
    docker push $ecr_uri:$ENVIRONMENT
    docker push $ecr_uri:latest
    
    # Deploy CloudFormation stack
    log "Deploying CloudFormation stack..."
    
    local template_file="aws-production-cloudformation.yaml"
    check_file "$template_file"
    
    # Prepare parameters
    local parameters=(
        "ParameterKey=Environment,ParameterValue=$ENVIRONMENT"
        "ParameterKey=ContainerImage,ParameterValue=$ecr_uri:$ENVIRONMENT"
        "ParameterKey=DBPassword,ParameterValue=$POSTGRES_PASSWORD"
        "ParameterKey=OpenRouterApiKey,ParameterValue=$OPENROUTER_API_KEY"
        "ParameterKey=JWTSecret,ParameterValue=$JWT_SECRET"
    )
    
    # Add optional parameters
    if [[ -n "${DOMAIN_NAME:-}" ]]; then
        parameters+=("ParameterKey=DomainName,ParameterValue=$DOMAIN_NAME")
    fi
    
    if [[ -n "${CERTIFICATE_ARN:-}" ]]; then
        parameters+=("ParameterKey=CertificateArn,ParameterValue=$CERTIFICATE_ARN")
    fi
    
    # Deploy stack
    aws cloudformation deploy \
        --template-file "$template_file" \
        --stack-name "$stack_name" \
        --parameter-overrides "${parameters[@]}" \
        --capabilities CAPABILITY_IAM \
        --region "$aws_region" \
        --no-fail-on-empty-changeset
    
    # Get stack outputs
    log "Getting deployment information..."
    local load_balancer_url=$(aws cloudformation describe-stacks \
        --stack-name "$stack_name" \
        --region "$aws_region" \
        --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerURL`].OutputValue' \
        --output text)
    
    success "AWS deployment completed successfully!"
    echo "Application URL: $load_balancer_url"
    echo "Stack Name: $stack_name"
    echo "Region: $aws_region"
}

# ================================
# Post-deployment Tasks
# ================================
post_deployment_tasks() {
    log "Running post-deployment tasks..."
    
    case "$DEPLOYMENT_TYPE" in
        "local")
            # Run database migrations
            log "Running database migrations..."
            docker-compose -f docker-compose.prod.yml exec openmanus python -m alembic upgrade head
            
            # Create default admin user
            log "Creating default admin user..."
            docker-compose -f docker-compose.prod.yml exec openmanus python scripts/create_admin.py
            ;;
        "render")
            log "Post-deployment tasks for Render should be configured in render-production.yaml"
            ;;
        "aws")
            log "Post-deployment tasks for AWS should be configured in ECS task definition"
            ;;
    esac
    
    success "Post-deployment tasks completed."
}

# ================================
# Cleanup
# ================================
cleanup() {
    log "Cleaning up temporary files..."
    
    # Remove temporary files
    rm -f /tmp/openmanus-*
    
    # Clean up Docker images (keep latest)
    docker image prune -f
    
    success "Cleanup completed."
}

# ================================
# Main Deployment Function
# ================================
main() {
    log "Starting OpenManus production deployment..."
    log "Deployment type: $DEPLOYMENT_TYPE"
    log "Environment: $ENVIRONMENT"
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Run deployment steps
    pre_deployment_checks
    build_application
    
    case "$DEPLOYMENT_TYPE" in
        "local")
            deploy_local
            ;;
        "render")
            deploy_render
            ;;
        "aws")
            deploy_aws
            ;;
        *)
            error "Unknown deployment type: $DEPLOYMENT_TYPE. Use 'local', 'render', or 'aws'."
            ;;
    esac
    
    post_deployment_tasks
    cleanup
    
    success "OpenManus deployment completed successfully!"
}

# ================================
# Usage Information
# ================================
usage() {
    echo "Usage: $0 [DEPLOYMENT_TYPE] [ENVIRONMENT]"
    echo
    echo "DEPLOYMENT_TYPE:"
    echo "  local   - Deploy using Docker Compose locally"
    echo "  render  - Deploy to Render platform"
    echo "  aws     - Deploy to AWS using CloudFormation"
    echo
    echo "ENVIRONMENT:"
    echo "  production (default)"
    echo "  staging"
    echo "  development"
    echo
    echo "Examples:"
    echo "  $0 local production"
    echo "  $0 render production"
    echo "  $0 aws production"
    echo
    echo "Environment Variables (for AWS deployment):"
    echo "  POSTGRES_PASSWORD - Database password"
    echo "  OPENROUTER_API_KEY - OpenRouter API key"
    echo "  JWT_SECRET - JWT secret key"
    echo "  AWS_REGION - AWS region (default: us-east-1)"
    echo "  DOMAIN_NAME - Custom domain name (optional)"
    echo "  CERTIFICATE_ARN - ACM certificate ARN (optional)"
}

# ================================
# Script Entry Point
# ================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Check for help flag
    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
        usage
        exit 0
    fi
    
    # Validate deployment type
    case "${1:-local}" in
        "local"|"render"|"aws")
            main "$@"
            ;;
        *)
            error "Invalid deployment type: ${1:-}. Use --help for usage information."
            ;;
    esac
fi

