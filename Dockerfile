FROM python:3.10-slim-buster

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTORCH_NO_CUDA=1
ENV TRANSFORMERS_OFFLINE=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK resources
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Copy project files
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Configure Gunicorn
ENV GUNICORN_CMD_ARGS="--workers=2 --timeout=120 --threads=4 --worker-class=gthread --worker-tmp-dir /dev/shm"

# Use gunicorn as production WSGI server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:main"]