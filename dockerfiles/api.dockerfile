FROM ghcr.io/astral-sh/uv:python3.12-alpine AS base

# Set working directory inside container
WORKDIR /app

# Copy dependency files first (better caching)
COPY uv.lock pyproject.toml ./

# Install dependencies (without project code first)
RUN uv sync --frozen --no-install-project

# Copy application source
COPY src ./src
COPY configs ./configs
COPY models ./models

# Install project itself
RUN uv sync --frozen

# Expose FastAPI port (important for cloud platforms)
EXPOSE 8000

# Run FastAPI app
ENTRYPOINT ["uv", "run", "uvicorn", "src.main_project.api:app", "--host", "0.0.0.0", "--port", "8000"]
