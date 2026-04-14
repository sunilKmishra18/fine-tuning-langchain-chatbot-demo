# Use official Python slim image to reduce image size
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Default Ollama host — override via docker-compose or docker run -e
    OLLAMA_HOST=http://ollama:11434

# Set working directory
WORKDIR /app

# Install dependencies before copying app code (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Run the app
CMD ["python", "main.py"]
