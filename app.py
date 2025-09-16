from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import io
import uvicorn
import os
import scipy.stats

app = FastAPI(title="Disaster Detection API")

# ----------------------
# Load model
# ----------------------
MODEL_PATH = "disaster_model.h5"  # Make sure this is the trained model file

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: '{MODEL_PATH}'")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully:", model.input_shape)

# ----------------------
# Class labels
# ----------------------
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

IMG_SIZE = 224  # EfficientNetB0 standard input size

# ----------------------
# Helper functions
# ----------------------
def preprocess_image(img_bytes):
    """Load image bytes → RGB → resize → numpy array → EfficientNet preprocess"""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")  # Always 3 channels
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)[None, ...]  # Shape: (1, 224, 224, 3)
    arr = preprocess_input(arr)
    return arr

def predict_image_from_bytes(img_bytes, threshold=0.7, entropy_threshold=1.0):
    arr = preprocess_image(img_bytes)
    
    preds = model.predict(arr, verbose=0)[0]
    entropy = float(scipy.stats.entropy(preds))
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx])
    label = CLASS_NAMES[top_idx]

    # If model is uncertain or low confidence → non-disaster
    if confidence < threshold or entropy > entropy_threshold:
        return {
            "label": "no_disaster_detected",
            "confidence": confidence,
            "entropy": entropy,
            "accepted": False
        }

    return {
        "label": label,
        "confidence": confidence,
        "entropy": entropy,
        "accepted": True
    }

# ----------------------
# FastAPI endpoint
# ----------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        result = predict_image_from_bytes(img_bytes)
        print("✅ Prediction result:", result)
        return JSONResponse(content=result)
    except Exception as e:
        print("❌ Error:", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})

# ----------------------
# Optional local run
# ----------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
