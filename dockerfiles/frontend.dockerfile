FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git curl && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements_frontend.txt .
COPY src/main_project/frontend.py ./frontend.py

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements_frontend.txt

EXPOSE 8501

# IMPORTANT: shell form so $PORT expands
CMD streamlit run frontend.py \
    --server.address=0.0.0.0 \
    --server.port=$PORT
