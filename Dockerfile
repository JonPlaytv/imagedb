FROM python:3.10-slim

WORKDIR /app

COPY . /app

# Install system packages
RUN apt-get update && apt-get install -y \
    ffmpeg libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

# Set environment variables
ENV PORT=8080

# Expose the port Render expects
EXPOSE 8080

# Start the Flask app
CMD ["python", "main.py"]
