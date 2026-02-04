FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git unzip ffmpeg libgl1 libglib2.0-0 build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

# Repo-required sources
ENV PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
ENV PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

RUN pip install --no-cache-dir -U pip setuptools wheel

# Install extras while blocking flash-attn/nvidia-pyindex
RUN pip install --no-cache-dir -e .
# RUN pip install --no-cache-dir --no-build-isolation -e ".[p3d]"
# RUN pip install --no-cache-dir --no-build-isolation -e ".[inference]"

# Hydra patch (repo requirement)
# RUN chmod +x ./patching/hydra && ./patching/hydra

# Server deps
RUN pip install --no-cache-dir fastapi uvicorn[standard] python-multipart pillow

ENV SAM3D_CHECKPOINTS_DIR=/checkpoints
ENV SAM3D_TAG=hf
ENV SAM3D_WORKDIR=/work
ENV SAM3D_COMPILE=0

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
