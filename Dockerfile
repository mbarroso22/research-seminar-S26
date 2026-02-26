FROM python:3.11-slim

WORKDIR /app

# System deps commonly needed for torchvision/pillow/opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Install python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy repo
COPY . /app

# Ensure we can import src/ and scripts/
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command: summarize (override as needed)
CMD ["python", "-m", "scripts.summarize_results"]