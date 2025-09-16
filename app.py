# app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
import scipy.stats

MODEL_PATH = "disaster_classifier.keras"  # Path to your saved .keras model
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

# Build the model architecture from scratch to ensure a clean state.
base_model = EfficientNetB0(
    include_top=False,
    weights=None,
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

# Now, load ONLY the weights from your fine-tuned model.
model.load_weights(MODEL_PATH)
model.trainable = False

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
        label = CLASS_NAMES[top_idx]

        # Use a confidence threshold to filter out low-confidence predictions
        threshold = 0.70
        
        # Check if the predicted label is a disaster class
        disaster_classes = ["damaged_buildings", "fallen_trees", "fire", "landslide", "flood"]
        
        # If the predicted label is a disaster class, apply a confidence check.
        # This prevents low-confidence disaster predictions from being returned.
        if label in disaster_classes and confidence < threshold:
            label = "no_disaster_detected"
        
        # You can also use an entropy check for more robust uncertainty handling.
        # Entropy measures the uncertainty of the prediction distribution.
        # A high entropy value indicates the model is unsure.
        entropy_threshold = 1.0  # Adjust as needed
        entropy = scipy.stats.entropy(preds)
        if entropy > entropy_threshold:
            label = "no_disaster_detected"
        
        # Format the final result
        result = {
            "predicted_class": label,
            "confidence": f"{confidence:.2%}",
            "all_probabilities": {cls: float(preds[i]) for i, cls in enumerate(CLASS_NAMES)}
        }
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.get("/")
async def root():
    return {"message": "Disaster Image Classifier API is running!"}