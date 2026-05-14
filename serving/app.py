"""
NysAdam: Model Serving API
===========================
Author: Hardik Tankaria
Date: May 2026
Description:
    FastAPI application for serving trained NysAdam models.
    Supports REST API endpoints for:
    - Image classification (CIFAR-10/100 models)
    - PINN evaluation (Burgers' equation)
    - Text generation (Transformer LM)
    - Model health/info endpoints

Usage:
    # Start server
    uvicorn serving.app:app --host 0.0.0.0 --port 8080

    # Or with gunicorn for production
    gunicorn serving.app:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080

API Endpoints:
    GET  /               → Health check
    GET  /info           → Model metadata
    POST /predict/image  → Classify an image
    POST /predict/pinn   → Evaluate PINN at (x, t) points
    POST /predict/text   → Generate text continuation
"""

import os
import sys
import io
import json
import base64
from typing import List, Optional

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

try:
    from fastapi import FastAPI, HTTPException, UploadFile, File
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "FastAPI not installed. Run: pip install fastapi uvicorn python-multipart"
    )

from nysadam.inference import load_checkpoint, predict_image, predict_pinn


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

CHECKPOINT_PATH = os.environ.get(
    "NYSADAM_CHECKPOINT",
    "results/checkpoints/cifar10_nysadam_best.pt",
)

# ═══════════════════════════════════════════════════════════════════
# App Initialization
# ═══════════════════════════════════════════════════════════════════

app = FastAPI(
    title="NysAdam Model Serving API",
    description="Serving models trained with NysAdam optimizer",
    version="0.1.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
model = None
metadata = None
device = None


@app.on_event("startup")
async def startup():
    """Load model on server start."""
    global model, metadata, device

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"WARNING: Checkpoint not found at {CHECKPOINT_PATH}")
        print("Set NYSADAM_CHECKPOINT environment variable to the correct path.")
        print("Server will start but /predict endpoints will return errors.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_checkpoint(CHECKPOINT_PATH, device)
    print(f"Model loaded: {metadata['model_class']} on {device}")
    print(f"  Optimizer: {metadata['optimizer']}")
    print(f"  Test accuracy: {metadata.get('test_accuracy', 'N/A')}")


# ═══════════════════════════════════════════════════════════════════
# Request/Response Schemas
# ═══════════════════════════════════════════════════════════════════

class PredictionResult(BaseModel):
    class_id: int
    class_name: str = ""
    confidence: float
    top5: List[dict] = []

class PINNRequest(BaseModel):
    x: List[float]
    t: List[float]

class PINNResponse(BaseModel):
    u: List[List[float]]
    x: List[float]
    t: List[float]

class TextRequest(BaseModel):
    prompt_tokens: List[int]
    max_new_tokens: int = 50
    temperature: float = 1.0

class TextResponse(BaseModel):
    generated_tokens: List[int]
    text: str = ""

class ModelInfo(BaseModel):
    model_class: str
    optimizer: str
    dataset: str
    test_loss: Optional[float]
    test_accuracy: Optional[float]
    training_time_sec: Optional[float]
    device: str
    status: str


# ═══════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════

@app.get("/")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if model is not None else "no_model",
        "model": metadata["model_class"] if metadata else None,
    }


@app.get("/info", response_model=ModelInfo)
async def info():
    """Return model metadata."""
    if metadata is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    return ModelInfo(
        model_class=metadata["model_class"],
        optimizer=metadata["optimizer"],
        dataset=metadata.get("dataset", "unknown"),
        test_loss=metadata.get("test_loss"),
        test_accuracy=metadata.get("test_accuracy"),
        training_time_sec=metadata.get("training_time_sec"),
        device=str(device),
        status="ready",
    )


@app.post("/predict/image", response_model=PredictionResult)
async def predict_image_endpoint(file: UploadFile = File(...)):
    """
    Classify an uploaded image using the trained model.

    Upload a JPEG/PNG image. Returns top-5 predicted classes with
    confidence scores.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    if metadata["model_class"] not in ("ResNet", "LogisticRegression"):
        raise HTTPException(
            status_code=400,
            detail=f"Image prediction not supported for {metadata['model_class']}",
        )

    try:
        from PIL import Image
        import torchvision.transforms as T
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="PIL/torchvision not available for image processing",
        )

    # Read and preprocess image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    transform = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    tensor = transform(img)

    # Class names
    cifar10_classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]
    cifar100_classes = None  # Would need a mapping file

    dataset = metadata.get("dataset", "")
    class_names = cifar10_classes if "cifar10" in dataset else cifar100_classes

    result = predict_image(model, tensor, class_names=class_names)
    return PredictionResult(**result)


@app.post("/predict/pinn", response_model=PINNResponse)
async def predict_pinn_endpoint(request: PINNRequest):
    """
    Evaluate PINN at given (x, t) grid points.

    Send arrays of x and t values. Returns the predicted solution u(x, t)
    on the full grid.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    if metadata["model_class"] != "PINN":
        raise HTTPException(
            status_code=400,
            detail=f"PINN prediction not supported for {metadata['model_class']}",
        )

    x = np.array(request.x)
    t = np.array(request.t)
    u = predict_pinn(model, x, t)

    return PINNResponse(
        u=u.tolist(),
        x=request.x,
        t=request.t,
    )


@app.post("/predict/text", response_model=TextResponse)
async def predict_text_endpoint(request: TextRequest):
    """
    Generate text continuation from a token sequence.

    Send a list of token IDs as prompt. Returns generated token IDs.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    if metadata["model_class"] != "TransformerLM":
        raise HTTPException(
            status_code=400,
            detail=f"Text generation not supported for {metadata['model_class']}",
        )

    input_ids = torch.tensor([request.prompt_tokens], device=device)
    generated = list(request.prompt_tokens)

    with torch.no_grad():
        for _ in range(request.max_new_tokens):
            logits = model(input_ids)
            next_logits = logits[0, -1, :] / request.temperature
            probs = F.softmax(next_logits, dim=0)
            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)
            input_ids = torch.tensor([generated[-256:]], device=device)

    # Decode if tokenizer available
    text = ""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        text = tokenizer.decode(generated)
    except ImportError:
        text = f"[{len(generated)} tokens generated, install transformers to decode]"

    return TextResponse(
        generated_tokens=generated,
        text=text,
    )
