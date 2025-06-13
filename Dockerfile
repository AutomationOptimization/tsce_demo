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
RUN pip install --no-cache-dir selfies tqdm

COPY . .
RUN chmod +x scripts/start-opensearch.sh
RUN ./scripts/start-opensearch.sh && \
    python tools/opensearch_index.py && pkill -f opensearch || true

CMD ["bash"]
