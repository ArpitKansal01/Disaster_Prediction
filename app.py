import os
import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load model once
MODEL_PATH = "disaster_classifier.keras"
model = tf.keras.models.load_model(MODEL_PATH)

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

app = FastAPI()

def predict_image(img_path):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    arr = np.array(img)[None, ...]
    arr = preprocess_input(arr)
    preds = model.predict(arr, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx])
    return CLASS_NAMES[top_idx], confidence

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    label, conf = predict_image(temp_path)
    os.remove(temp_path)
    return {"label": label, "confidence": conf}
