# Use optimized Python base image
FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \          # Suppress TensorFlow info messages
    OMP_NUM_THREADS=1 \               # Single thread for OpenMP
    TF_NUM_INTEROP_THREADS=1 \        # Single thread for TensorFlow inter-op
    TF_NUM_INTRAOP_THREADS=1          # Single thread for TensorFlow intra-op

# Install system dependencies (added curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \                            # Required for healthcheck
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Create non-root user
RUN useradd -m -u 1001 appuser && \
    mkdir -p /app/models && \
    chown -R appuser:appuser /app

# Copy application files
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Optimize TensorFlow for CPU deployment
ENV TF_ENABLE_ONEDNN_OPTS=1 \         # Enable CPU optimizations
    TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"  # CPU-specific XLA flags

# Remove GPU-related dependencies from requirements
# (Ensure requirements.txt doesn't include tensorflow-gpu or similar)

# Expose port
EXPOSE 7860

# Healthcheck (requires curl installed)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s \
    CMD curl --fail http://localhost:7860/health || exit 1

# Optimized entrypoint with adaptive workers
ENTRYPOINT ["gunicorn", "--worker-class", "gthread", \
           "--threads", "4", "--bind", "0.0.0.0:7860", \
           "--timeout", "120", "--keep-alive", "65", \
           "--max-requests", "1000", "--max-requests-jitter", "50", \
           "--access-logfile", "-", "--error-logfile", "-", \
           "app:app"]
