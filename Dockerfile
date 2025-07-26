FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package management
RUN pip install uv

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Install dependencies with pre-release support
RUN uv pip install --system --prerelease=allow -r requirements.txt

# Install playwright browsers
RUN playwright install --with-deps chromium

# Copy application code
COPY . .

# Create config directory if it doesn't exist
RUN mkdir -p config

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["python", "main.py"]
