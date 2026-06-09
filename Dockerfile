FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml requirements.txt ./
COPY config ./config
COPY proto ./proto
COPY src ./src
COPY tests ./tests

RUN python -m pip install --upgrade pip \
    && python -m pip install \
        "numpy>=1.26.0,<2.0.0" \
        "fastapi>=0.115.0" \
        "uvicorn[standard]>=0.30.0" \
        "pydantic>=2.7.0" \
        "pydantic-settings>=2.3.0" \
        "pyyaml>=6.0.1" \
        "httpx>=0.27.0" \
        "pytest>=8.0.0" \
        "pytest-asyncio>=0.23.0" \
    && python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu "torch==2.2.2+cpu" \
    && python -m pip install -e ".[dev]" --no-deps

EXPOSE 9000

CMD ["python", "-m", "src"]
