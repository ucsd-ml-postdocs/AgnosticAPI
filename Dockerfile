# Use the official Ubuntu 22.04 as the base image
FROM python:3.11

COPY ./requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y libhdf5-dev libhdf5-serial-dev
RUN apt-get install -y libhdf5-dev
RUN pip install Cython --upgrade
RUN pip install setuptools --upgrade
RUN pip install wheel --upgrade
RUN pip install h5py
# # Install pip
# RUN apt-get install -y python3-pip
# #RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

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
