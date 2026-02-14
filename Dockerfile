FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# OS deps in a stable layer
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates curl build-essential \
      libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda (simple + deterministic)
RUN curl -fsSL -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  && bash /tmp/miniconda.sh -b -p /opt/conda \
  && rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# Install mamba (faster, more reliable solving)
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda install -y -n base -c conda-forge mamba && conda clean -afy

# ---- Dependency layer (changes rarely) ----
# Copy only env spec first so this layer is cached
ENV CONDA_OVERRIDE_CUDA=12.1
COPY environment.yml /app/environment.yml

# Create the env in a fixed prefix (easier to activate)
RUN --mount=type=cache,target=/opt/conda/pkgs \
    mamba env create -f /app/environment.yml -p /opt/conda/envs/sam3d \
    && conda clean -afy

# Make the env the default python
ENV PATH=/opt/conda/envs/sam3d/bin:$PATH

# ---- App layer (changes often) ----
COPY pyproject.toml pyproject.toml

# Install your package without re-resolving deps (keep pyproject deps empty/minimal)
RUN /opt/conda/envs/sam3d/bin/pip install --no-cache-dir -e ".[server]"
# RUN pip install -e ".[server]"

COPY sam3d_objects/ sam3d_objects/
COPY server.py server.py
COPY inference.py inference.py

EXPOSE 8000

# These should be mounted from NAS at runtime
ENV SAM3D_CONFIG="/checkpoints/pipeline.yaml"
ENV LIDRA_SKIP_INIT=1

CMD ["conda", "run", "--no-capture-output", "-p", "/opt/conda/envs/sam3d", \
     "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
