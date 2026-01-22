FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base

WORKDIR /app

# Copy dependency files
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY LICENSE LICENSE
COPY configs/ configs/

RUN uv sync --frozen --no-install-project

COPY src src/
COPY data data/

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "src/main_project/train.py"]
#run with docker run --name experiment_new \--env-file .env \train:new