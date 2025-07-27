#!/bin/bash

# OpenManus Local Deployment Script
# This script sets up OpenManus for local development using Docker Compose

set -e

echo "🚀 Starting OpenManus Local Deployment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your API keys before continuing."
    echo "   Required: OPENAI_API_KEY, OPENROUTER_API_KEY, ANTHROPIC_API_KEY"
    read -p "Press Enter after updating .env file..."
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs workspace nginx/ssl scripts

# Generate SSL certificates for local development
if [ ! -f nginx/ssl/localhost.crt ]; then
    echo "🔐 Generating SSL certificates for local development..."
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout nginx/ssl/localhost.key \
        -out nginx/ssl/localhost.crt \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
fi

# Create nginx configuration for development
cat > nginx/nginx.dev.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream openmanus_api {
        server openmanus:8000;
    }

    upstream openmanus_mcp {
        server mcp-server:8081;
    }

    server {
        listen 80;
        server_name localhost;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl;
        server_name localhost;

        ssl_certificate /etc/nginx/ssl/localhost.crt;
        ssl_certificate_key /etc/nginx/ssl/localhost.key;

        location / {
            proxy_pass http://openmanus_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /mcp/ {
            proxy_pass http://openmanus_mcp/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /ws {
            proxy_pass http://openmanus_api;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
        }
    }
}
EOF

# Create database initialization script
cat > scripts/init-db.sql << 'EOF'
-- OpenManus Database Initialization
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create tables for OpenManus
CREATE TABLE IF NOT EXISTS agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    config JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS flows (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'created',
    config JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    flow_id UUID REFERENCES flows(id),
    agent_id UUID REFERENCES agents(id),
    input_data JSONB,
    output_data JSONB,
    status VARCHAR(50) DEFAULT 'pending',
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mcp_tools (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    schema JSONB,
    agent_id UUID REFERENCES agents(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_agents_name ON agents(name);
CREATE INDEX IF NOT EXISTS idx_flows_type ON flows(type);
CREATE INDEX IF NOT EXISTS idx_flows_status ON flows(status);
CREATE INDEX IF NOT EXISTS idx_executions_flow_id ON executions(flow_id);
CREATE INDEX IF NOT EXISTS idx_executions_status ON executions(status);
CREATE INDEX IF NOT EXISTS idx_mcp_tools_agent_id ON mcp_tools(agent_id);
EOF

# Build and start services
echo "🔨 Building Docker images..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service health
echo "🔍 Checking service health..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ OpenManus API is healthy"
else
    echo "❌ OpenManus API is not responding"
fi

if curl -f http://localhost:8081/health > /dev/null 2>&1; then
    echo "✅ MCP Server is healthy"
else
    echo "❌ MCP Server is not responding"
fi

# Show service URLs
echo ""
echo "🎉 OpenManus is now running locally!"
echo ""
echo "📍 Service URLs:"
echo "   • API Server: http://localhost:8000"
echo "   • API Server (HTTPS): https://localhost"
echo "   • MCP Server: http://localhost:8081"
echo "   • Frontend: http://localhost:3001"
echo "   • Database: localhost:5432"
echo "   • Redis: localhost:6379"
echo ""
echo "📊 Monitoring:"
echo "   • Logs: docker-compose logs -f"
echo "   • Status: docker-compose ps"
echo ""
echo "🛑 To stop: docker-compose down"
echo "🗑️  To clean: docker-compose down -v --rmi all"

