FROM ghcr.io/astral-sh/uv:python3.12-bookworm

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY LICENSE LICENSE
COPY configs/ configs/

COPY src src/

RUN uv sync --frozen --no-install-project



# COPY data data/

# RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "src/main_project/cloud_train.py"]
