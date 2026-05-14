# ═══════════════════════════════════════════════════════════════════
# NysAdam Model Serving - Dockerfile
# ═══════════════════════════════════════════════════════════════════
# Author: Hardik Tankaria | Date: May 2026
#
# Build:  docker build -t nysadam-serving .
# Run:    docker run -p 8080:8080 -v $(pwd)/results:/app/results nysadam-serving
#
# For GPU support:
#   docker run --gpus all -p 8080:8080 -v $(pwd)/results:/app/results nysadam-serving
# ═══════════════════════════════════════════════════════════════════

FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (CPU-only torch for serving — smaller image)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    torchvision --index-url https://download.pytorch.org/whl/cpu \
    numpy \
    scipy \
    scikit-learn \
    fastapi \
    uvicorn \
    python-multipart \
    Pillow

# Copy project
COPY . /app/

# Install package
RUN pip install --no-cache-dir -e .

# Environment
ENV NYSADAM_CHECKPOINT=results/checkpoints/cifar10_nysadam_best.pt
ENV PORT=8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/')"

# Run server
EXPOSE ${PORT}
CMD uvicorn serving.app:app --host 0.0.0.0 --port ${PORT}
