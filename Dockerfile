FROM python:3.11-slim

# Install system dependencies for building wheels like RDKit
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libboost-all-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash"]
