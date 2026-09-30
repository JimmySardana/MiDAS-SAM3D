# SAM3D FastAPI Server — Deployment Guide

A containerised 3-D segmentation inference service built on [SAM3D](https://github.com/...), served via FastAPI/Uvicorn, scheduled on a SLURM cluster with `sbatch`, and exposed to the outside world through a two-hop SSH tunnel.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Repository Structure](#repository-structure)
4. [Step 1 — Build & Push the Docker Image](#step-1--build--push-the-docker-image)
5. [Step 2 — Deploy the Server via SLURM](#step-2--deploy-the-server-via-slurm)
6. [Step 3 — Open the SSH Tunnel](#step-3--open-the-ssh-tunnel)
7. [API Reference](#api-reference)
8. [Configuration Reference](#configuration-reference)
9. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
Your Machine :8001
      │  (local SSH tunnel)
      ▼
Login Node (amperecontrol1) :18001
      │  (reverse tunnel, opened by sbatch job)
      ▼
Compute Node :8001
      │
      ▼
Docker Container (sam3d) :8000
      │
      ▼
FastAPI / Uvicorn  ←  SAM3D Inference Pipeline
```

The SLURM job:
1. Copies model checkpoints from the NAS if not already cached.
2. Starts the Docker container with GPU access.
3. Opens a **reverse SSH tunnel** from the compute node back to the login node, forwarding the container port.

---

## Prerequisites

| Requirement | Details |
|---|---|
| Docker (rootless) | Available on compute nodes |
| SLURM | `sbatch` command available |
| SSH key | `~/.ssh/id_ed25519_cluster` — passwordless access to `amperecontrol1` |
| Registry access | `pc3163.igd.fraunhofer.de:4567` |
| NAS credentials | `~/smbcreds` file on the compute node for checkpoint download |
| CUDA 12.1 | Required on compute nodes |

Before deploying, replace every `ldapUser` placeholder in `scripts/sam3d_sbatch.sh` with your LDAP username. This includes the login account and the checkpoint cache path. Use the same username in the `scp` and SSH commands shown below.

---

## Repository Structure

```
.
├── Dockerfile          # Two-stage CUDA build
├── scripts/
│   ├── build_push.sh   # Build & push image to registry
│   └── sam3d_sbatch.sh # SLURM job: run container + reverse tunnel
├── server.py           # FastAPI application
├── inference.py        # Inference wrapper
├── environment.yml     # Conda environment
├── requirements.txt    # Pip dependencies
└── pyproject.toml      # Package definition
```

---

## Step 1 — Build & Push the Docker Image

Run the following script **from your local machine or a build node** with Docker access:

```bash
bash scripts/build_push.sh
```

This will:
1. Prompt you to log in to the private registry at `pc3163.igd.fraunhofer.de:4567`.
2. Push the `sam3d` image to the registry.

> **Note:** Make sure you have built the image before pushing:
> ```bash
> docker build -t pc3163.igd.fraunhofer.de:4567/sam3d .
> ```

The Dockerfile uses a **two-stage build**:
- **Stage 1 (builder):** Installs Miniforge, creates the `sam3d` conda environment, and installs all pip/package dependencies.
- **Stage 2 (runtime):** Copies only the conda environment and application code into a lean CUDA runtime image, keeping the final image size minimal.

---

## Step 2 — Deploy the Server via SLURM

After updating `ldapUser` in the script, copy it from your local machine to the cluster login node.
Then connect to the login node and submit the job **on the cluster**:

```bash
sbatch sam3d_sbatch.sh
```

### What the job does

| Phase | Action |
|---|---|
| Checkpoint sync | Mounts the NAS via SMB and copies `sam3d_checkpoint.zip` to `/opt/cache/ldapUser/` (skipped if already present) |
| Container start | Launches the `sam3d` Docker container with GPU access and mounts the checkpoint directory to `/checkpoints` |
| Reverse tunnel | Opens an SSH reverse tunnel: `compute:8001 → amperecontrol1:18001` |
| Health loop | Monitors both processes; exits cleanly if either dies |
| Cleanup | On `SIGTERM`/job cancellation, stops the tunnel and the container gracefully |

### SLURM resource allocation

| Parameter | Value |
|---|---|
| GPUs | 1 per node |
| CPUs | 64 per GPU |
| Memory | 128 GB |
| Wall time | 60 minutes |

### Monitor the job

```bash
# Check job status
squeue -u $USER

# Follow live logs
tail -f slurm-sam3d-cluster-<JOB_ID>.out
```

---

## Step 3 — Open the SSH Tunnel

Once the SLURM job is running, open the final tunnel **from your local machine**:

```bash
ssh -N -L 0.0.0.0:8001:127.0.0.1:18001 ldapUser@amperecontrol1
```

This forwards `localhost:8001` on your machine through the login node to the container running on the compute node.

The server is now reachable at:

```
http://localhost:8001
```

---

## API Reference

### `GET /health`

Returns the current server status and whether the model is loaded.

**Response**
```json
{
  "ok": true,
  "model_loaded": true
}
```

---

### `POST /infer/glb`

Runs 3-D inference on an image and a segmentation mask, returning a `.glb` 3-D model file.

**Request** — `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `image` | `file` | Input RGB image (PNG/JPEG) |
| `mask` | `file` | Segmentation mask image |

**Response**

| Property | Value |
|---|---|
| Content-Type | `model/gltf-binary` |
| Body | Binary `.glb` file |
| Filename | `result.glb` |

**Example (curl)**

```bash
curl -X POST http://localhost:8001/infer/glb \
  -F "image=@input.png" \
  -F "mask=@mask.png" \
  --output result.glb
```

**Error responses**

| Code | Cause |
|---|---|
| `500` | Inference pipeline failed |
| `500` | GLB export failed |
| `500` | Model config file not found at startup |

---

## Configuration Reference

The following environment variables control the container's behaviour:

| Variable | Default | Description |
|---|---|---|
| `SAM3D_CONFIG` | `/checkpoints/pipeline.yaml` | Path to the Hydra pipeline config |
| `LIDRA_SKIP_INIT` | `1` | Skips optional initialisation steps |
| `SPARSE_ATTN_BACKEND` | `sdpa` | Sparse attention backend |
| `ATTN_BACKEND` | `sdpa` | Attention backend |

The checkpoint directory is mounted from the compute node at runtime:
```
/opt/cache/ldapUser/sam3d_checkpoint  →  /checkpoints  (inside container)
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `SAM3D_CONFIG not found` | Ensure the checkpoint directory was correctly copied from the NAS and the volume mount is correct in `sam3d_sbatch.sh` |
| Tunnel not connecting | Verify `~/.ssh/id_ed25519_cluster` has passwordless access to `amperecontrol1` |
| Container fails to start | Check `slurm-sam3d-cluster-<JOB_ID>.err` for Docker errors; ensure the image was pushed successfully |
| `LOGIN_PORT` already in use | Another job may be using port `18001` on the login node; adjust `LOGIN_PORT` in `sam3d_sbatch.sh` and the destination port in the Step 3 SSH command |
| Job times out before inference | Increase `--time` in `sam3d_sbatch.sh` |
