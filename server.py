import os
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# IMPORTANT: keep this import if the repo requires side effects/registrations.
# (Some configs rely on modules being imported before Hydra instantiation.)
import sam3d_objects  # noqa: F401

# If you're using your extracted core:
# from sam3d_infer.core import Inference
#
# Otherwise, import the repo's inference (adjust if your path differs):
# NOTE: importing notebook code in a server is okay if you've pruned viz out.
from notebook.inference import Inference  # type: ignore


APP_NAME = "sam3d-infer"


class InferRequest(BaseModel):
    input_path: str = Field(..., description="Path to input pointcloud/scene file inside the container")
    prompt: str = Field(..., description="Text prompt, e.g. 'chair'")
    output_dir: Optional[str] = Field(None, description="Directory to write outputs (defaults to /outputs)")
    # Optional: if your Inference supports other prompt types
    # box: Optional[list[float]] = None
    # mask_path: Optional[str] = None


class InferResponse(BaseModel):
    ok: bool
    output_path: str
    elapsed_ms: int
    meta: Dict[str, Any] = {}


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


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest) -> InferResponse:
    global _model
    if _model is None:
        # fallback if startup loading disabled
        _model = _load_model()

    in_path = Path(req.input_path)
    if not in_path.exists():
        raise HTTPException(status_code=400, detail=f"input_path does not exist: {in_path}")

    out_dir = Path(req.output_dir or "/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = uuid.uuid4().hex[:10]
    out_path = out_dir / f"sam3d_{run_id}.npz"

    t0 = time.time()

    # ---- Call into your inference API here ----
    # You MUST adapt this call to whatever your Inference class provides.
    #
    # Common patterns:
    #   result = _model.run(input_path=str(in_path), prompt=req.prompt, output_path=str(out_path))
    #   result = _model(input_path=str(in_path), prompt=req.prompt)  # then you save
    #
    # Below is a generic example:
    try:
        result = _model.run(
            input_path=str(in_path),
            prompt=req.prompt,
            output_path=str(out_path),
        )
    except AttributeError:
        # If your Inference doesn't have .run(), try a different call shape:
        try:
            result = _model(
                input_path=str(in_path),
                prompt=req.prompt,
                output_path=str(out_path),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference call failed: {type(e).__name__}: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {type(e).__name__}: {e}")

    elapsed_ms = int((time.time() - t0) * 1000)

    # Keep response stable even if result is None / complex
    meta: Dict[str, Any] = {}
    if isinstance(result, dict):
        # shallow copy only JSON-ish things
        for k, v in result.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                meta[k] = v

    if not out_path.exists():
        # Some implementations write different files. If so, return what you actually wrote.
        # Still treat this as a test failure.
        raise HTTPException(status_code=500, detail=f"Expected output not created at: {out_path}")

    return InferResponse(ok=True, output_path=str(out_path), elapsed_ms=elapsed_ms, meta=meta)
