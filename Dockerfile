FROM python:3.11-slim

WORKDIR /app

# System deps needed for torchvision/Pillow/opencv in many cases
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt /app/requirements.txt

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the repo
COPY . /app

# Ensure Python can import "src" when running scripts
ENV PYTHONPATH=/app

# Default command (you can override in docker run)
CMD ["python", "-m", "scripts.summarize_results"]