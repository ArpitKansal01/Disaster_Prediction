from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import scipy.stats
import io
import uvicorn

app = FastAPI()

# Load model
IMG_SIZE = 224
MODEL_PATH = "disaster_classifier.keras"
model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
print("✅ Model loaded with input shape:", model.input_shape)

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

def predict_image_from_bytes(img_bytes, threshold=0.7, entropy_threshold=1.0):
    # Open the image from bytes and convert to RGB
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    
    arr = tf.keras.utils.img_to_array(img)[None, ...]  # shape: (1, 224, 224, 3)
    arr = preprocess_input(arr)
    
    preds = model.predict(arr, verbose=0)[0]
    entropy = float(scipy.stats.entropy(preds))
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx])
    label = CLASS_NAMES[top_idx]

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

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        result = predict_image_from_bytes(img_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Optional: run locally
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
