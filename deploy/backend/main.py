import io, json, os
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from model import load_backend_model, predict_topk
from gcs_utils import download_from_gcs

# NEW: GCS model location
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "dannmodelweights")
MODEL_BLOB = os.getenv("MODEL_BLOB", "Dann_visualOnly_twoStage.pth")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "/tmp/Dann_visualOnly_twoStage.pth")

CLASSNAMES_PATH = os.getenv("CLASSNAMES_PATH", "class_names.json")
TOPK = int(os.getenv("TOPK", "1"))
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "ViT-B-32")
CLIP_PRETRAINED = os.getenv("CLIP_PRETRAINED", "openai")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app = FastAPI(title="CLIP+DANN Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOWED_ORIGINS == ["*"] else ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
B = {}

@app.on_event("startup")
def startup():
    # 1) download weights from GCS into /tmp
    ckpt_path = download_from_gcs(MODEL_BUCKET, MODEL_BLOB, LOCAL_MODEL_PATH)

    # 2) load class names
    with open(CLASSNAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    # 3) load models
    clip_model, dann, preprocess, device_t = load_backend_model(
        checkpoint_path=ckpt_path,
        class_names=class_names,
        clip_model_name=CLIP_MODEL_NAME,
        clip_pretrained=CLIP_PRETRAINED,
        device=device,
    )

    B.update(
        clip_model=clip_model,
        dann=dann,
        preprocess=preprocess,
        device=device_t,
        class_names=class_names,
    )

@app.get("/health")
def health():
    return {"status": "ok", "device": str(B["device"]), "num_classes": len(B["class_names"])}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail="Upload JPEG/PNG/WebP")

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Bad image")

    results = predict_topk(
        B["clip_model"], B["dann"], B["preprocess"], B["device"], img, B["class_names"], TOPK
    )
    return {"topk": results}