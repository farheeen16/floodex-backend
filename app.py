from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
import torch.nn as nn
import numpy as np
import rasterio
import cv2
from torchvision import transforms, models
import tempfile
import os

# ==========================================
# CONFIG
# ==========================================
MODEL_PATH = "best_texture_vgg16.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ["non_flooded", "flooded"]   # MUST match training
NUM_CLASSES = len(class_names)

# ==========================================
# LOAD MODEL (MATCHES TRAINING ARCHITECTURE)
# ==========================================
model = models.vgg16(pretrained=False)

model.classifier = nn.Sequential(
    nn.Linear(25088, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(512, NUM_CLASSES)
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

softmax = nn.Softmax(dim=1)

# ==========================================
# FASTAPI APP
# ==========================================
app = FastAPI(title="Flood Texture Classification API")

# ==========================================
# INFERENCE FUNCTION (YOUR LOGIC)
# ==========================================
def predict_tif(path):
    with rasterio.open(path) as src:
        img = src.read([1, 2, 3]).astype(np.float32)

    img = np.transpose(img, (1, 2, 0))

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx**2 + gy**2)

    mag = (mag - mag.mean()) / (mag.std() + 1e-6)
    mag = np.stack([mag, mag, mag], axis=2)

    x = torch.from_numpy(mag).permute(2, 0, 1)
    x = transforms.Resize((224, 224))(x)
    x = x.unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        probs = softmax(model(x))[0].cpu().numpy()

    return {
        class_names[i]: float(probs[i])
        for i in range(NUM_CLASSES)
    }

# ==========================================
# ROUTES
# ==========================================
@app.get("/")
def health():
    return {"status": "API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".tif"):
        raise HTTPException(status_code=400, detail="Only .tif files are supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        prediction = predict_tif(tmp_path)
    finally:
        os.remove(tmp_path)

    return {
        "filename": file.filename,
        "prediction": prediction
    }
