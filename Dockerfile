# syntax=docker/dockerfile:1
#
# Mira in remote mode: photos are uploaded through the web UI (or copied into the
# /photos volume), indexed and analysed on the server.
#
#   docker build -t mira .                          # CPU
#   docker build -t mira:gpu --build-arg EXTRA=gpu . # NVIDIA GPU (run with --gpus all)
#   docker run -p 8765:8765 -e VT_WEB__PASSWORD=… -v vt-data:/data -v vt-photos:/photos mira
#
# See docker-compose.yml for a setup with Ollama (the vision model) next to it.

# A fixed base image (tag and digest), so a build does not take whatever the tag points to
# today. To update, take a newer 3.12.x-slim tag and its digest from Docker Hub.
ARG PYTHON_IMAGE=python:3.12.15-slim@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704

FROM ${PYTHON_IMAGE} AS build
ARG EXTRA=cpu
# insightface is built from source: it needs a C++ compiler
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.11.32
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev --extra ${EXTRA}
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --extra ${EXTRA}

FROM ${PYTHON_IMAGE}
# libgomp: onnxruntime / scikit-learn threads; libglib: OpenCV
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1000 vision \
 && mkdir -p /data /photos \
 && chown vision:vision /data /photos
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    VT_MODE=remote \
    VT_REMOTE__UPLOAD_DIR=/photos \
    VT_PATHS__DATA_DIR=/data \
    VT_FACES__MODEL_ROOT=/data \
    VT_WEB__HOST=0.0.0.0 \
    VT_WEB__PORT=8765 \
    VT_VLM__BASE_URL=http://ollama:11434
USER vision
WORKDIR /data
VOLUME ["/data", "/photos"]
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/healthz', timeout=4)"
ENTRYPOINT ["vision-tagger"]
CMD ["serve", "--migrate"]
