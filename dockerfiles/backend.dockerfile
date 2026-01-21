FROM ghcr.io/astral-sh/uv:python3.12-bookworm

# Set platform to amd64 for compatibility (or use ARM-specific PyTorch)
ARG TARGETPLATFORM
RUN echo "Building for platform: $TARGETPLATFORM"

# Install system dependencies needed for PyTorch
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libopenblas-dev \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY LICENSE LICENSE
COPY configs/ configs/

# First sync without source code
RUN uv sync --frozen --no-install-project

# Copy source code
COPY src src/

RUN mkdir /app

# Copy data
COPY data app/data/
# Copy models into image
COPY models app/models/


# Final sync with source code
RUN uv sync --frozen

# Expose port for FastAPI
EXPOSE 8000

# Change entrypoint to run your FastAPI app instead of training
#ENTRYPOINT ["uv", "run", "src/main_project/backend.py"]



ENTRYPOINT ["uv", "run", "uvicorn", "src.main_project.backend:app", "--host", "0.0.0.0", "--port", "8000"]
