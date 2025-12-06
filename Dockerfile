FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data
COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/
COPY inputs/ ./inputs/

# Create outputs directory
RUN mkdir -p outputs

# Set Python path to include src directory
ENV PYTHONPATH=/app/src

# Default command (can be overridden)
CMD ["python", "src/main.py", "--help"]

