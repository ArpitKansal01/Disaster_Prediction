# app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf

MODEL_PATH = "disaster_model.h5"  # path to your saved model
IMG_SIZE = 224
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

# Load model once
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

app = FastAPI(title="Disaster Image Classifier API")

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Converts image to RGB, resizes, and preprocesses for EfficientNetB0."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image)[None, ...]  # shape: (1, H, W, 3)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return arr

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file)
        arr = preprocess_image(img)
        preds = model.predict(arr, verbose=0)[0]
        top_idx = int(np.argmax(preds))
        confidence = float(preds[top_idx])
        result = {
            "predicted_class": CLASS_NAMES[top_idx],
            "confidence": f"{confidence:.2%}",
            "all_probabilities": {cls: float(preds[i]) for i, cls in enumerate(CLASS_NAMES)}
        }
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.get("/")
async def root():
    return {"message": "Disaster Image Classifier API is running!"}
