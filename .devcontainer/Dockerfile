FROM mcr.microsoft.com/devcontainers/base:ubuntu

# Installeer C++ tools
RUN apt-get update && apt-get install -y \
    cmake \
    gcc \
    g++ \
    make \
    gdb \
    libopencv-dev \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    && apt-get clean
