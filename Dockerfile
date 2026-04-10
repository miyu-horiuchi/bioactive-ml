FROM python:3.11-slim AS base

WORKDIR /app

# System deps for RDKit
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (CPU-only torch for smaller image)
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    torch-geometric \
    fastapi uvicorn[standard] pydantic && \
    pip install --no-cache-dir \
    rdkit-pypi ripser persim pandas numpy scikit-learn \
    matplotlib networkx requests fair-esm

# Copy application code
COPY *.py ./
COPY data/ ./data/

# Copy model checkpoints if they exist
COPY checkpoints/ ./checkpoints/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
