FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for audio, video capture, and Bluetooth
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    alsa-utils \
    pulseaudio-utils \
    mpv \
    libgl1 \
    libglib2.0-0 \
    bluez \
    dbus \
    network-manager \
    iputils-ping \
    arping \
    && rm -rf /var/lib/apt/lists/*

# Install python packages
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    opencv-python-headless \
    numpy \
    jinja2 \
    python-multipart \
    gTTS \
    edge-tts \
    httpx \
    beautifulsoup4 \
    scapy

# Copy source code
COPY . .

# Expose backend port
EXPOSE 8050

# Run FastAPI
CMD ["python", "main.py"]
