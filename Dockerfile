# CUDA build and runtime image for CudaForge.
#
# Multi-stage: the builder carries the full toolkit (nvcc, headers, CMake),
# while the runtime image keeps only the CUDA runtime libraries. The devel image
# is several gigabytes larger, and none of that is needed to execute the result.
#
# Requires on the host:
#   * an NVIDIA driver
#   * the NVIDIA Container Toolkit
#   * `--gpus all` at run time
#
# This image has not been built or run for this repository — the development
# host is Apple Silicon, where nvidia/cuda images are the wrong architecture and
# no GPU can be passed through. See docs/environment.md.

ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04

# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      ninja-build \
      git \
      ca-certificates \
      python3 \
      python3-dev \
      python3-pip \
      python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/cudaforge

# Dependency manifests first, so the expensive pip layer is cached across source
# edits. Without this split, every code change reinstalls PyTorch.
COPY pyproject.toml setup.py ./
COPY python/cudaforge/__init__.py python/cudaforge/__init__.py
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install torch numpy

COPY . .

# Narrow this to the target architecture for a faster build and a smaller
# binary; the default is wide so an unmodified image runs on most cards.
# Starts at 80: the BF16 kernels need compute capability 8.0, and below it
# bfloat16 is emulated rather than unsupported — slower than the FP16 path it
# replaces, which is worse than a build failure because nothing reports it.
ARG CUDA_ARCHITECTURES="80;86;89;90"
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

RUN cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCUDAFORGE_ENABLE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
    && cmake --build build --parallel

RUN python3 -m pip install --no-build-isolation ".[train,serve,dev]"

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user. A container that writes checkpoints into a mounted
# volume as root leaves files the host user cannot delete.
RUN useradd --create-home --shell /bin/bash cudaforge
WORKDIR /opt/cudaforge

COPY --from=builder /usr/lib/python3/dist-packages /usr/lib/python3/dist-packages
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder --chown=cudaforge:cudaforge /opt/cudaforge /opt/cudaforge

USER cudaforge
ENV PYTHONPATH=/opt/cudaforge/python:/opt/cudaforge

EXPOSE 8000

# Reports whether the GPU is visible and which implementation path is active, so
# a misconfigured container fails its health check rather than silently serving
# from the reference path.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"

CMD ["python3", "-c", "from cudaforge.ops import backend_report; print(backend_report())"]
