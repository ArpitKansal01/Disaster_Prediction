from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import scipy.stats
import io
import os

app = FastAPI()

# ----------------------
# Load model
# ----------------------
MODEL_PATH = "new_model.h5"
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

IMG_SIZE = model.input_shape[1]  # ensures exact match with model

# ----------------------
# Helper functions
# ----------------------
def preprocess_image(img_bytes):
    """Convert uploaded bytes to model-ready numpy array"""
    img = Image.open(io.BytesIO(img_bytes))

    # Force RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)

    # Safety check: ensure 3 channels
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    elif arr.shape[-1] != 3:
        arr = arr[:, :, :3]  # crop extra channels if any

    arr = arr[None, ...]  # add batch dimension
    arr = preprocess_input(arr)
    return arr

def predict_image_from_bytes(img_bytes, threshold=0.7, entropy_threshold=1.0):
    """Predict class and return structured result"""
    arr = preprocess_image(img_bytes)
    preds = model.predict(arr, verbose=0)[0]
    
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx])
    entropy = float(scipy.stats.entropy(preds))
    label = CLASS_NAMES[top_idx]

    # Apply threshold checks
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
        if len(img_bytes) == 0:
            return JSONResponse(status_code=400, content={"error": "Empty file uploaded"})

        try:
            result = predict_image_from_bytes(img_bytes)
        except UnidentifiedImageError:
            return JSONResponse(status_code=400, content={"error": "Uploaded file is not a valid image"})

        print("✅ Prediction result:", result)
        return JSONResponse(content=result)

    except Exception as e:
        print("❌ Error:", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})

# ----------------------
# Optional local run
# ----------------------
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
