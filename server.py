# server.py
import base64
import io
import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
import numpy as np

# Import sam3d-objects inference utilities (as per official quick start)
import sys
sys.path.append("notebook")
from inference import Inference, load_image  # type: ignore

APP_WORKDIR = Path(os.environ.get("SAM3D_WORKDIR", "/work")).resolve()
CHECKPOINTS_DIR = Path(os.environ.get("SAM3D_CHECKPOINTS_DIR", "/checkpoints")).resolve()
TAG = os.environ.get("SAM3D_TAG", "hf")
COMPILE = os.environ.get("SAM3D_COMPILE", "0") == "1"

CONFIG_PATH = CHECKPOINTS_DIR / TAG / "pipeline.yaml"

app = FastAPI(title="sam3d-objects server")

_infer: Optional[Inference] = None


def _ensure_dirs():
    (APP_WORKDIR / "jobs").mkdir(parents=True, exist_ok=True)


def _load_model():
    global _infer
    if _infer is not None:
        return

    if not CONFIG_PATH.exists():
        raise RuntimeError(
            f"pipeline.yaml not found at {CONFIG_PATH}. "
            f"Mount checkpoints so that {CHECKPOINTS_DIR}/{TAG}/pipeline.yaml exists."
        )

    # Create inference object once (GPU heavy)
    _infer = Inference(str(CONFIG_PATH), compile=COMPILE)


def _read_upload_to_pil(up: UploadFile) -> Image.Image:
    data = up.file.read()
    if not data:
        raise HTTPException(400, f"Empty upload: {up.filename}")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img
    except Exception as e:
        raise HTTPException(400, f"Failed to decode image {up.filename}: {e}")


def _pil_mask_to_binary(mask_pil: Image.Image) -> np.ndarray:
    """
    Accepts:
      - L / RGB / RGBA mask image
    Produces:
      - HxW bool mask where True means object.
    Convention: non-zero alpha or non-zero luminance => object.
    """
    if mask_pil.mode == "RGBA":
        a = np.array(mask_pil.split()[-1])
        return a > 0
    if mask_pil.mode != "L":
        mask_pil = mask_pil.convert("L")
    m = np.array(mask_pil)
    return m > 0


@app.on_event("startup")
def startup():
    _ensure_dirs()
    _load_model()


@app.get("/health")
def health():
    ok = CONFIG_PATH.exists()
    return {
        "ok": bool(ok),
        "tag": TAG,
        "compile": COMPILE,
        "config_path": str(CONFIG_PATH),
        "workdir": str(APP_WORKDIR),
    }


@app.post("/reconstruct")
async def reconstruct(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    seed: int = Form(42),
):
    """
    Multipart request:
      - image: RGB(A) image (png/jpg)
      - mask:  binary mask png (white/alpha = object)
      - seed:  int
    Output:
      - saves splat.ply to /work/jobs/<job_id>/splat.ply
      - returns {job_id, files}
    """
    if _infer is None:
        _load_model()

    job_id = uuid.uuid4().hex[:12]
    job_dir = APP_WORKDIR / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Decode uploads
    img_pil = _read_upload_to_pil(image).convert("RGBA")  # safe default
    mask_pil = _read_upload_to_pil(mask)

    # Convert to the formats inference expects
    # In their notebooks, load_image returns an RGBA image and mask is separate.
    # We'll pass numpy arrays; Inference in notebook code generally handles PIL/np.
    img_rgba = np.array(img_pil)  # HxWx4 uint8
    mask_bin = _pil_mask_to_binary(mask_pil).astype(np.uint8)  # HxW {0,1}

    # Run model
    try:
        output = _infer(img_rgba, mask_bin, seed=seed)
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")

    # Save gaussian splat ply (matches quick start)
    # output["gs"] is expected to have save_ply
    files = []
    try:
        ply_path = job_dir / "splat.ply"
        output["gs"].save_ply(str(ply_path))
        files.append("splat.ply")
    except Exception as e:
        raise HTTPException(500, f"Failed to save ply: {e}")

    # Optional: save debug inputs
    try:
        img_pil.save(job_dir / "input.png")
        Image.fromarray(mask_bin * 255).convert("L").save(job_dir / "mask.png")
        files += ["input.png", "mask.png"]
    except Exception:
        pass

    return JSONResponse(
        {
            "ok": True,
            "job_id": job_id,
            "files": files,
            "download_base": f"/download/{job_id}",
        }
    )


@app.get("/download/{job_id}/{filename}")
def download(job_id: str, filename: str):
    p = (APP_WORKDIR / "jobs" / job_id / filename).resolve()
    if not str(p).startswith(str((APP_WORKDIR / "jobs").resolve())):
        raise HTTPException(400, "Invalid path")
    if not p.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(str(p), filename=filename)
