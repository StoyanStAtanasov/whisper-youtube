# Use an official Python runtime as a parent image
#FROM python:3.9-slim-buster
FROM pytorch/pytorch:latest

# # Install the NVIDIA runtime
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     nvidia-cuda-toolkit \
#     && rm -rf /var/lib/apt/lists/*

# Set the working directory is /workspace in the pytorch container
#WORKDIR /app

# install git so we can install the requirements
RUN apt update
RUN apt install -y git

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Run the command to start the application
CMD ["python", "run.py"]