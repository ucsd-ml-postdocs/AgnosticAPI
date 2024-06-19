# Use the official Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
    pkg-config \
    libhdf5-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Install required Python packages
RUN pip3 install tensorflow==2.12
RUN pip3 install fastapi uvicorn tqdm Pillow nibabel matplotlib

# Set the working directory
#WORKDIR /app
WORKDIR /AgnosticAPI/app/server

# Copy application code to the container
#COPY . /AgnosticAPI/app/server
# Copy the entire AgnosticAPI directory to the WORKDIR
COPY ./app/server /AgnosticAPI/app/server

# Expose the port
EXPOSE 8000

# Default command to run when starting the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
