# app.py
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import scipy.stats
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

# -----------------------------
# 1️⃣ Model Setup
# -----------------------------
MODEL_PATH = "disaster_classifier.keras"
IMG_SIZE = 224
CONF_THRESHOLD = 0.7       # Minimum confidence to accept prediction
ENTROPY_THRESHOLD = 1.0    # Maximum entropy to accept prediction

# Class labels
CLASS_NAMES = [
    "damaged_buildings",
    "fallen_trees",
    "fire",
    "landslide",
    "flood",
    "non_damage_building",
    "non_damage_forest",
    "non_disaster",
    "sea",
]

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: '{MODEL_PATH}'")

model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

app = FastAPI(title="Safe Disaster Image Classifier API")

# -----------------------------
# 2️⃣ Utility Function
# -----------------------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convert image to RGB, resize and preprocess for EfficientNet."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image)[None, ...]  # Shape: (1, H, W, 3)
    arr = preprocess_input(arr)
    return arr

def predict_safe(arr: np.ndarray, threshold=CONF_THRESHOLD, entropy_threshold=ENTROPY_THRESHOLD):
    """Predict with confidence and entropy checks."""
    preds = model.predict(arr, verbose=0)[0]
    pred_entropy = float(scipy.stats.entropy(preds))
    
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx])
    label = CLASS_NAMES[top_idx]

    if confidence < threshold or pred_entropy > entropy_threshold:
        return "no_disaster_detected", confidence, pred_entropy

    return label, confidence, pred_entropy

# -----------------------------
# 3️⃣ API Endpoints
# -----------------------------
@app.get("/")
async def root():
    return {"message": "Safe Disaster Image Classifier API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), threshold: float = CONF_THRESHOLD, entropy_threshold: float = ENTROPY_THRESHOLD):
    try:
        img = Image.open(file.file)
        arr = preprocess_image(img)
        label, confidence, pred_entropy = predict_safe(arr, threshold, entropy_threshold)
        
        result = {
            "predicted_class": label,
            "confidence": f"{confidence:.2%}",
            "entropy": pred_entropy
        }
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
