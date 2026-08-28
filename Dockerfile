# Legacy builder compatible: Docker 18.09 / docker-compose 1.22.
# CPU-only HTTP orchestration and image processing; no CANN/NPU runtime is needed.
ARG BASE_IMAGE=python:3.11-slim-bookworm
FROM ${BASE_IMAGE}

ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" -r /app/requirements.txt
COPY app /app/app

EXPOSE 18080
ENTRYPOINT []
CMD ["python3", "-m", "app.main"]
