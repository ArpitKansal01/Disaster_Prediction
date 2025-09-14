from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import scipy.stats
import io

app = FastAPI()

# Load model
MODEL_PATH = "disaster_classifier.keras"
model = tf.keras.models.load_model(MODEL_PATH)

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
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
    arr = np.array(img)[None, ...]
    arr = preprocess_input(arr)
    preds = model.predict(arr, verbose=0)[0]
    entropy = scipy.stats.entropy(preds)
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx])
    label = CLASS_NAMES[top_idx]
    
    if confidence < threshold or entropy > entropy_threshold:
        return {
    "label": "no_disaster_detected",
    "confidence": float(confidence),
    "entropy": float(entropy),
    "accepted": False
}


    return {
    "label": str(label),
    "confidence": float(confidence),  # Convert from np.float32 to Python float
    "entropy": float(entropy),
    "accepted": bool(True)
}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        result = predict_image_from_bytes(img_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
