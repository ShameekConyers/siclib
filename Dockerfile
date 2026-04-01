FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libopenblas-dev liblapacke-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . .

RUN pip install pybind11 numpy pytest scipy setuptools wheel
RUN pip install -e .
RUN pytest tests/ -v
