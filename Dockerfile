FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git curl wget build-essential \
    pkg-config libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy and modify requirements
COPY requirements.txt .
RUN sed -i 's/crawl4ai>0.6.3,<0.7.dev0/crawl4ai==0.6.3/g' requirements.txt
RUN sed -i 's/uv==/uv>=/g' requirements.txt || true

# Install dependencies one by one for better error visibility
RUN pip install --no-cache-dir --pre -r requirements.txt

# Install playwright
RUN python -m playwright install chromium
RUN python -m playwright install-deps chromium

COPY . .
RUN mkdir -p config

EXPOSE 8000
CMD ["python", "main.py"]
