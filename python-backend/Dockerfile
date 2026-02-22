# Coronary RWS Analyser v2 - Production Dockerfile
# Multi-stage build for smaller image size

# =============================================================================
# Stage 1: Build dependencies
# =============================================================================
FROM python:3.11-slim-bullseye as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies to /app/.local
RUN pip install --no-cache-dir --prefix=/app/.local -r requirements.txt

# =============================================================================
# Stage 2: Runtime image
# =============================================================================
FROM python:3.11-slim-bullseye

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Copy Python packages from builder
COPY --from=builder /app/.local /app/.local
ENV PATH=/app/.local/bin:$PATH
ENV PYTHONPATH=/app/.local/lib/python3.11/site-packages:$PYTHONPATH

# Copy application code
COPY app/ ./app/

# Create data directory
RUN mkdir -p /app/data && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with uvicorn
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
