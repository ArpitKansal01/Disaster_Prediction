from PIL import Image
import numpy as np
import io
import tensorflow as tf

MODEL_PATH = "my_model.keras"
IMG_SIZE = 224

# Load the model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded")

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

def preprocess_image(image_bytes):
    # Open image and convert to RGB (3 channels)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert to numpy array and preprocess for EfficientNet
    arr = np.array(img)[None, ...]  # Shape: (1, 224, 224, 3)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return arr

def predict(image_bytes):
    arr = preprocess_image(image_bytes)
    probs = model.predict(arr, verbose=0)[0]
    # Return class probabilities
    return {cls: float(probs[i]) for i, cls in enumerate(CLASS_NAMES)}
