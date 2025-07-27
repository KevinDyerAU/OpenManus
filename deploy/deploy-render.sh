#!/bin/bash

# OpenManus Render Deployment Script
# This script deploys OpenManus to Render using their CLI

set -e

echo "🚀 Starting OpenManus Render Deployment..."

# Check if Render CLI is installed
if ! command -v render &> /dev/null; then
    echo "❌ Render CLI is not installed. Installing..."
    curl -fsSL https://cli.render.com/install | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Check if user is logged in to Render
if ! render auth whoami &> /dev/null; then
    echo "🔐 Please log in to Render..."
    render auth login
fi

# Validate render.yaml exists
if [ ! -f render.yaml ]; then
    echo "❌ render.yaml not found. Please ensure you're in the project root directory."
    exit 1
fi

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
    echo "Please set these variables and run the script again:"
    echo "export OPENAI_API_KEY='your-key-here'"
    echo "export OPENROUTER_API_KEY='your-key-here'"
    echo "export ANTHROPIC_API_KEY='your-key-here'"
    exit 1
fi

# Create Render service configuration
echo "📝 Preparing Render deployment..."

# Build and push Docker image to Render
echo "🔨 Building Docker image for Render..."
docker build -f Dockerfile.improved --target render -t openmanus:render .

# Deploy to Render
echo "🚀 Deploying to Render..."
render deploy

# Set environment variables in Render
echo "🔧 Setting environment variables..."
render env set OPENAI_API_KEY="$OPENAI_API_KEY" --service-name=openmanus-api
render env set OPENROUTER_API_KEY="$OPENROUTER_API_KEY" --service-name=openmanus-api
render env set ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" --service-name=openmanus-api

# Wait for deployment to complete
echo "⏳ Waiting for deployment to complete..."
sleep 60

# Get service URL
SERVICE_URL=$(render services list --format json | jq -r '.[] | select(.name=="openmanus-api") | .serviceDetails.url')

if [ "$SERVICE_URL" != "null" ] && [ -n "$SERVICE_URL" ]; then
    echo "✅ Deployment successful!"
    echo ""
    echo "🌐 Service URL: $SERVICE_URL"
    echo "🔍 Health Check: $SERVICE_URL/health"
    echo ""
    echo "📊 Monitoring:"
    echo "   • Logs: render logs --service-name=openmanus-api"
    echo "   • Status: render services list"
    echo "   • Dashboard: https://dashboard.render.com"
    
    # Test the deployment
    echo "🧪 Testing deployment..."
    if curl -f "$SERVICE_URL/health" > /dev/null 2>&1; then
        echo "✅ Health check passed!"
    else
        echo "⚠️  Health check failed. Check logs: render logs --service-name=openmanus-api"
    fi
else
    echo "❌ Failed to get service URL. Check deployment status:"
    echo "render services list"
    exit 1
fi

echo ""
echo "🎉 OpenManus is now deployed on Render!"
echo ""
echo "📚 Next steps:"
echo "   • Configure custom domain (optional)"
echo "   • Set up monitoring and alerts"
echo "   • Configure backup strategies"
echo "   • Review security settings"

