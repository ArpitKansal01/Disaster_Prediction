# app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0

MODEL_PATH = "disaster_model.h5"
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

# Load the entire model from the file, which includes architecture and weights.
# The previous attempt to build and load weights separately is failing.
# This approach is generally more reliable for models saved with tf.keras.Model.save().
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    # If loading the entire model fails, try the rebuild-and-load-weights approach as a fallback.
    print(f"Failed to load entire model: {e}. Attempting to rebuild and load weights.")
    # Rebuild the same model architecture from scratch
    base_model = EfficientNetB0(
        include_top=False,
        weights=None, # This is crucial to prevent loading imagenet weights
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    model = Model(inputs, outputs)

    # Load only the weights from the saved file.
    model.load_weights(MODEL_PATH)

app = FastAPI(title="Disaster Image Classifier API")

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Converts image to RGB, resizes, and preprocesses for EfficientNetB0."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image)[None, ...]
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