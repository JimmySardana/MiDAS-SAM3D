# MiDAS SAM 3D Objects service

This repository packages a pruned version of [Meta's SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) as a GPU-backed FastAPI service. Send an image and a mask for one object; the service returns a binary GLB model of that object.

The Docker image contains the inference code and server, **not the model checkpoints**. Download the SAM 3D Objects checkpoints separately and mount them when starting the container.

## Requirements

- A Linux host with an NVIDIA GPU, a working NVIDIA Container Toolkit, and Docker with BuildKit support. The image uses CUDA 12.1.
- Access to the [SAM 3D Objects checkpoints on Hugging Face](https://huggingface.co/facebook/sam-3d-objects). Request access and authenticate before downloading; see [Meta's checkpoint setup instructions](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md#getting-checkpoints).
- An image and a corresponding object mask of the same dimensions. The mask must encode the selected object in its **alpha channel** (opaque object, transparent background); the server reads that channel to determine the mask.

## Get the checkpoints

After access is approved, download the official SAM 3D Objects checkpoint bundle using Meta's instructions. For example, from the repository root on a Linux host:

```bash
pip install 'huggingface-hub[cli]<1.0'
hf auth login
hf download --repo-type model --local-dir checkpoints/download --max-workers 1 facebook/sam-3d-objects
mv checkpoints/download/checkpoints checkpoints/hf
```

The directory mounted into the container must contain `pipeline.yaml` at its top level, alongside the checkpoint files and directories referenced by that configuration. With the example above, that path is `checkpoints/hf/pipeline.yaml`. Checkpoints are ignored by Git and excluded from the Docker build context.

## Run locally

Run these commands from the repository root in a Bash shell:

```bash
docker build -t sam3d .
docker run --rm --gpus all \
  -p 8000:8000 \
  -v "$(pwd)/checkpoints/hf:/checkpoints:ro" \
  sam3d
```

The server loads the model during startup; it needs the mounted checkpoint directory before it can accept requests. The default configuration path is `/checkpoints/pipeline.yaml` and can be changed with `SAM3D_CONFIG`.

Check that it started:

```bash
curl http://localhost:8000/health
```

A ready server returns `{"ok":true,"model_loaded":true}`. FastAPI also provides interactive API documentation at `http://localhost:8000/docs`.

## Generate a GLB

`POST /infer/glb` accepts `multipart/form-data` with two file fields: `image` and `mask`. It returns the GLB bytes with media type `model/gltf-binary` and filename `result.glb`.

```bash
curl --fail --show-error \
  -F "image=@test/inference/images/image.png" \
  -F "mask=@test/inference/images/mask.png" \
  http://localhost:8000/infer/glb \
  -o result.glb
```

Inference uses a fixed seed of `42`. The request can take substantially longer than the health check. The example client in [test/inference/test.ipynb](test/inference/test.ipynb) uses port `8001`, which is convenient when forwarding the cluster service to a local port; change its `SERVER_URL` for your setup.

## Fraunhofer cluster deployment

The scripts in `scripts/` build and push the image to the Fraunhofer registry, then run it as a Slurm job with GPU access and a reverse SSH tunnel. The job copies the checkpoint archive from NAS if it is not already cached on the compute node. See the [Fraunhofer deployment guide](docs/fraunhofer-deployment.md) for checkpoint upload, configuration, submission, and access instructions.

## Repository layout

| Path | Purpose |
| --- | --- |
| `server.py` | FastAPI endpoints and GLB response |
| `inference.py` | SAM 3D Objects inference wrapper and image loading |
| `sam3d_objects/` | Pruned upstream model and pipeline code |
| `Dockerfile`, `environment.yml`, `requirements.txt` | CUDA image and Python dependencies |
| `scripts/` | Registry build/push and Slurm deployment |
| `test/inference/` | Example client notebook and sample inputs |

## Troubleshooting

- **Startup reports `SAM3D_CONFIG not found`:** verify that the mounted directory contains `pipeline.yaml` directly under `/checkpoints`, or set `SAM3D_CONFIG` to the correct container path.
- **The whole image is reconstructed:** check that the mask PNG has a transparent background. An ordinary opaque RGB or grayscale PNG becomes fully opaque when loaded and will be treated as a full-image mask.
- **GPU or CUDA errors:** confirm GPU access inside Docker with `--gpus all` and check host driver compatibility with CUDA 12.1.
