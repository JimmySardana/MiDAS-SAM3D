import os
import trimesh
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from inference import Inference, load_image, load_mask
from pydantic import BaseModel, Field

# IMPORTANT: keep this import if the repo requires side effects/registrations.
# (Some configs rely on modules being imported before Hydra instantiation.)
import sam3d_objects  # noqa: F401

# If you're using your extracted core:
# from sam3d_infer.core import Inference
#
# Otherwise, import the repo's inference (adjust if your path differs):
# NOTE: importing notebook code in a server is okay if you've pruned viz out.


APP_NAME = "sam3d-infer"


app = FastAPI(title=APP_NAME)

_model: Optional[Inference] = None


def _get_env_path(name: str, default: str) -> str:
    v = os.environ.get(name, default)
    return v


def _load_model() -> Inference:
    """
    Loads your inference pipeline once.

    Required env vars depend on how your Inference class is constructed.
    Common pattern for this repo:
      - SAM3D_CONFIG=/checkpoints/pipeline.yaml (NAS-mounted)
    """
    config_file = _get_env_path("SAM3D_CONFIG", "/checkpoints/pipeline.yaml")

    if not Path(config_file).exists():
        raise RuntimeError(f"SAM3D_CONFIG not found: {config_file}")

    model = Inference(
        config_file,
        compile=False
    )

    return model


@app.on_event("startup")
def startup() -> None:
    """
    Load model at startup so /infer is fast and we fail fast if deps are missing.
    If you prefer lazy load, comment this out and load on first request.
    """
    global _model
    _model = _load_model()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "model_loaded": _model is not None,
    }

@app.post("/infer/glb")
async def infer_glb(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
):
    global _model
    if _model is None:
        _model = _load_model()

    image_np = load_image(await image.read())
    mask_np = load_mask(await mask.read())

    try:
        output = _model(image_np, mask_np, seed=42)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e

    glb_bytes = extract_glb_bytes(output["glb"])

    return Response(
        content=glb_bytes,
        media_type="model/gltf-binary",
        headers={"Content-Disposition": 'attachment; filename="result.glb"'},
    )

def extract_glb_bytes(glb: trimesh.Trimesh) -> bytes:
    if glb is None:
        raise HTTPException(
            status_code=500,
            detail=f"No trimesh/GLB object found in output.",
        )

    try:
        glb_bytes = glb.export(file_type="glb")
        return glb_bytes
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export GLB: {e}") from e