#FROM --platform=linux/amd64 ubuntu:16.04

FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app/app

RUN export PATH=$PATH:/app/app/c3d-1.4.0-Linux-gcc64/bin/
