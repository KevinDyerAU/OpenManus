#!/bin/bash

# OpenManus AWS Deployment Script
# This script deploys OpenManus to AWS using ECS Fargate, RDS, and ElastiCache

set -e

# Configuration
ENVIRONMENT_NAME=${ENVIRONMENT_NAME:-openmanus-prod}
AWS_REGION=${AWS_REGION:-us-east-1}
STACK_NAME=${STACK_NAME:-$ENVIRONMENT_NAME-infrastructure}

echo "🚀 Starting OpenManus AWS Deployment..."
echo "   Environment: $ENVIRONMENT_NAME"
echo "   Region: $AWS_REGION"
echo "   Stack: $STACK_NAME"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is not installed. Please install AWS CLI first."
    exit 1
fi

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "❌ jq is not installed. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install jq
    else
        sudo apt-get update && sudo apt-get install -y jq
    fi
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS credentials not configured. Please run 'aws configure' first."
    exit 1
fi

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "📋 AWS Account ID: $AWS_ACCOUNT_ID"

# Check for required environment variables
echo "🔍 Checking environment variables..."
REQUIRED_VARS=("OPENAI_API_KEY" "OPENROUTER_API_KEY" "ANTHROPIC_API_KEY")
MISSING_VARS=()

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        MISSING_VARS+=("$var")
    fi
done

if [ ${#MISSING_VARS[@]} -ne 0 ]; then
    echo "❌ Missing required environment variables:"
    printf '   • %s\n' "${MISSING_VARS[@]}"
    echo ""
    echo "Please set these variables and run the script again."
    exit 1
fi

# Create ECR repository if it doesn't exist
ECR_REPO_NAME="openmanus"
echo "🏗️  Setting up ECR repository..."

if ! aws ecr describe-repositories --repository-names $ECR_REPO_NAME --region $AWS_REGION &> /dev/null; then
    echo "📦 Creating ECR repository..."
    aws ecr create-repository \
        --repository-name $ECR_REPO_NAME \
        --region $AWS_REGION \
        --image-scanning-configuration scanOnPush=true
else
    echo "✅ ECR repository already exists"
fi

# Get ECR login token and login to Docker
echo "🔐 Logging into ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build and push Docker image
ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME"
IMAGE_TAG="latest"

echo "🔨 Building Docker image..."
docker build -f Dockerfile.improved --target aws -t $ECR_URI:$IMAGE_TAG .

echo "📤 Pushing image to ECR..."
docker push $ECR_URI:$IMAGE_TAG

# Store secrets in AWS Secrets Manager
echo "🔐 Setting up secrets in AWS Secrets Manager..."

SECRETS=(
    "openmanus/secret-key:$(openssl rand -base64 32)"
    "openmanus/openai-api-key:$OPENAI_API_KEY"
    "openmanus/openrouter-api-key:$OPENROUTER_API_KEY"
    "openmanus/anthropic-api-key:$ANTHROPIC_API_KEY"
)

for secret in "${SECRETS[@]}"; do
    SECRET_NAME=$(echo $secret | cut -d: -f1)
    SECRET_VALUE=$(echo $secret | cut -d: -f2)
    
    if aws secretsmanager describe-secret --secret-id $SECRET_NAME --region $AWS_REGION &> /dev/null; then
        echo "🔄 Updating secret: $SECRET_NAME"
        aws secretsmanager update-secret \
            --secret-id $SECRET_NAME \
            --secret-string $SECRET_VALUE \
            --region $AWS_REGION > /dev/null
    else
        echo "📝 Creating secret: $SECRET_NAME"
        aws secretsmanager create-secret \
            --name $SECRET_NAME \
            --secret-string $SECRET_VALUE \
            --region $AWS_REGION > /dev/null
    fi
done

# Deploy CloudFormation stack
echo "☁️  Deploying CloudFormation stack..."

# Check if stack exists
if aws cloudformation describe-stacks --stack-name $STACK_NAME --region $AWS_REGION &> /dev/null; then
    echo "🔄 Updating existing stack..."
    OPERATION="update-stack"
else
    echo "📝 Creating new stack..."
    OPERATION="create-stack"
fi

# Deploy the stack
aws cloudformation $OPERATION \
    --stack-name $STACK_NAME \
    --template-body file://aws-cloudformation.yaml \
    --parameters ParameterKey=EnvironmentName,ParameterValue=$ENVIRONMENT_NAME \
    --capabilities CAPABILITY_IAM \
    --region $AWS_REGION

echo "⏳ Waiting for CloudFormation stack to complete..."
aws cloudformation wait stack-${OPERATION//-stack/}-complete \
    --stack-name $STACK_NAME \
    --region $AWS_REGION

# Get stack outputs
echo "📋 Getting stack outputs..."
OUTPUTS=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --region $AWS_REGION \
    --query 'Stacks[0].Outputs')

LOAD_BALANCER_URL=$(echo $OUTPUTS | jq -r '.[] | select(.OutputKey=="LoadBalancerURL") | .OutputValue')
DATABASE_ENDPOINT=$(echo $OUTPUTS | jq -r '.[] | select(.OutputKey=="DatabaseEndpoint") | .OutputValue')
CACHE_ENDPOINT=$(echo $OUTPUTS | jq -r '.[] | select(.OutputKey=="CacheEndpoint") | .OutputValue')

# Update secrets with database and cache URLs
echo "🔄 Updating database and cache connection secrets..."
aws secretsmanager create-secret \
    --name "openmanus/database-url" \
    --secret-string "postgresql://openmanus:$(aws secretsmanager get-random-password --password-length 32 --exclude-characters '"@/\' --output text --query RandomPassword)@$DATABASE_ENDPOINT:5432/openmanus" \
    --region $AWS_REGION > /dev/null 2>&1 || \
aws secretsmanager update-secret \
    --secret-id "openmanus/database-url" \
    --secret-string "postgresql://openmanus:$(aws secretsmanager get-random-password --password-length 32 --exclude-characters '"@/\' --output text --query RandomPassword)@$DATABASE_ENDPOINT:5432/openmanus" \
    --region $AWS_REGION > /dev/null

aws secretsmanager create-secret \
    --name "openmanus/redis-url" \
    --secret-string "redis://$CACHE_ENDPOINT:6379/0" \
    --region $AWS_REGION > /dev/null 2>&1 || \
aws secretsmanager update-secret \
    --secret-id "openmanus/redis-url" \
    --secret-string "redis://$CACHE_ENDPOINT:6379/0" \
    --region $AWS_REGION > /dev/null

# Update ECS task definition with correct image URI
echo "🔧 Updating ECS task definition..."
sed "s|ACCOUNT_ID|$AWS_ACCOUNT_ID|g; s|REGION|$AWS_REGION|g" aws-ecs-task-definition.json > /tmp/task-definition.json

# Register new task definition
TASK_DEFINITION_ARN=$(aws ecs register-task-definition \
    --cli-input-json file:///tmp/task-definition.json \
    --region $AWS_REGION \
    --query 'taskDefinition.taskDefinitionArn' \
    --output text)

echo "📝 Registered task definition: $TASK_DEFINITION_ARN"

# Update ECS service
echo "🔄 Updating ECS service..."
aws ecs update-service \
    --cluster $ENVIRONMENT_NAME-cluster \
    --service $ENVIRONMENT_NAME-service \
    --task-definition $TASK_DEFINITION_ARN \
    --region $AWS_REGION > /dev/null

echo "⏳ Waiting for service to stabilize..."
aws ecs wait services-stable \
    --cluster $ENVIRONMENT_NAME-cluster \
    --services $ENVIRONMENT_NAME-service \
    --region $AWS_REGION

# Test the deployment
echo "🧪 Testing deployment..."
if curl -f "$LOAD_BALANCER_URL/health" > /dev/null 2>&1; then
    echo "✅ Health check passed!"
else
    echo "⚠️  Health check failed. Checking service status..."
    aws ecs describe-services \
        --cluster $ENVIRONMENT_NAME-cluster \
        --services $ENVIRONMENT_NAME-service \
        --region $AWS_REGION \
        --query 'services[0].events[0:3]'
fi

# Clean up temporary files
rm -f /tmp/task-definition.json

echo ""
echo "🎉 OpenManus is now deployed on AWS!"
echo ""
echo "📍 Service Information:"
echo "   • Load Balancer URL: $LOAD_BALANCER_URL"
echo "   • Health Check: $LOAD_BALANCER_URL/health"
echo "   • Database Endpoint: $DATABASE_ENDPOINT"
echo "   • Cache Endpoint: $CACHE_ENDPOINT"
echo "   • ECR Repository: $ECR_URI"
echo ""
echo "📊 Monitoring:"
echo "   • ECS Console: https://$AWS_REGION.console.aws.amazon.com/ecs/home?region=$AWS_REGION#/clusters/$ENVIRONMENT_NAME-cluster"
echo "   • CloudWatch Logs: https://$AWS_REGION.console.aws.amazon.com/cloudwatch/home?region=$AWS_REGION#logsV2:log-groups/log-group/%2Fecs%2F$ENVIRONMENT_NAME"
echo "   • CloudFormation: https://$AWS_REGION.console.aws.amazon.com/cloudformation/home?region=$AWS_REGION#/stacks/stackinfo?stackId=$STACK_NAME"
echo ""
echo "🛠️  Management Commands:"
echo "   • View logs: aws logs tail /ecs/$ENVIRONMENT_NAME --follow --region $AWS_REGION"
echo "   • Scale service: aws ecs update-service --cluster $ENVIRONMENT_NAME-cluster --service $ENVIRONMENT_NAME-service --desired-count <count> --region $AWS_REGION"
echo "   • Delete stack: aws cloudformation delete-stack --stack-name $STACK_NAME --region $AWS_REGION"
echo ""
echo "📚 Next steps:"
echo "   • Configure custom domain and SSL certificate"
echo "   • Set up CloudWatch alarms and monitoring"
echo "   • Configure backup strategies for RDS"
echo "   • Review security groups and IAM policies"
echo "   • Set up CI/CD pipeline for automated deployments"

