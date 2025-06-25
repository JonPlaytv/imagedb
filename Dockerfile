FROM python:3.10-slim

WORKDIR /app

COPY . /app

# Install system dependencies needed for image processing and git
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies and download spaCy model
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

# Expose the port for Flask
ENV PORT=8080
EXPOSE 8080

# Start the app
CMD ["python", "main.py"]
