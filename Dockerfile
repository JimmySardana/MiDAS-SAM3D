# ============================================================
# Stage 1: Build environment
# ============================================================
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# OS build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates curl build-essential \
      libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (includes mamba, no TOS issues)
RUN curl -fsSL -o /tmp/miniforge.sh \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh
ENV PATH=/opt/conda/bin:$PATH

# Conda env (cached unless environment.yml changes)
ENV CONDA_OVERRIDE_CUDA=12.1
COPY environment.yml .
RUN --mount=type=cache,target=/opt/conda/pkgs \
    mamba env create -f environment.yml -p /opt/conda/envs/sam3d \
    && conda clean -afy

ENV PATH=/opt/conda/envs/sam3d/bin:$PATH

# Pip deps (cached unless requirements.txt changes)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Install the package (non-editable)
COPY pyproject.toml .
COPY sam3d_objects/ sam3d_objects/
RUN pip install --no-deps --no-cache-dir .

# ============================================================
# Stage 2: Runtime (smaller image)
# ============================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Runtime OS deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire conda env from builder
COPY --from=builder /opt/conda/envs/sam3d /opt/conda/envs/sam3d
ENV PATH=/opt/conda/envs/sam3d/bin:$PATH

# App code
COPY server.py .
COPY inference.py .

# Runtime config
ENV SAM3D_CONFIG="/checkpoints/pipeline.yaml" \
    LIDRA_SKIP_INIT=1 \
    SPARSE_ATTN_BACKEND="sdpa" \
    ATTN_BACKEND="sdpa"

EXPOSE 8000

# No conda run needed — PATH already points to the env
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]