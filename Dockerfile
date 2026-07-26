# ============================================================================
# Multi-stage Dockerfile for the Bangladeshi Gov Services RAG System
# ============================================================================
FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies required by ChromaDB and lxml
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libxml2-dev \
        libxslt1-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Application stage
# ---------------------------------------------------------------------------
FROM base AS app

WORKDIR /app

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY crawler.py .
COPY ingest.py .
COPY rag_engine.py .
COPY app.py .

# Create directories for persistent data
RUN mkdir -p /app/chroma_db /app/data

# Expose Streamlit port
EXPOSE 8501

# Health check – Streamlit serves on 8501
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command: run the Streamlit app
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
