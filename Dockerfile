# Use Python 3.11 as base image
FROM python:3.11.5-slim

# Set working directory
WORKDIR /app

# Install system dependencies for matplotlib and other packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p uploads static/eda_plots

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_DEBUG=False
ENV PYTHONUNBUFFERED=1

# Expose port (default 5000, but configurable via PORT env var)
EXPOSE ${PORT:-5000}

# Run the application using gunicorn with proper timeouts for heavy ML operations
CMD sh -c "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 2 --timeout 120 --worker-class sync --max-requests 1000 --max-requests-jitter 100 app:app"
