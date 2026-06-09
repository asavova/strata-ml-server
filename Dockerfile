FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY . /app

RUN python -m pip install pip==26.1.2 \
    && python -m pip install -r requirements.txt \
    && python -m pip install -e . --no-deps

EXPOSE 9000

CMD ["python", "-m", "src"]
