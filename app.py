import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input  # pyright: ignore[reportMissingImports]
import scipy.stats

# Load model
MODEL_PATH = "disaster_classifier.keras"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: '{MODEL_PATH}'")

model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

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

def predict_image(img_path, threshold=0.7, entropy_threshold=1.0):
    # Load image
    img = Image.open(img_path)

    # Ensure RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize
    img = img.resize((224, 224))

    # Convert to array and preprocess
    arr = np.array(img)
    if arr.shape[-1] != 3:
        arr = np.stack([arr]*3, axis=-1)  # Just in case
    arr = arr[None, ...]  # Add batch dimension
    arr = preprocess_input(arr)

    # Predict
    preds = model.predict(arr, verbose=0)[0]

    # Entropy
    entropy = float(scipy.stats.entropy(preds))
    print(f"Prediction entropy: {entropy:.3f}")

    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx])
    label = CLASS_NAMES[top_idx]

    if confidence < threshold or entropy > entropy_threshold:
        print(f"⚠️ No disaster detected ")
        return "no_disaster_detected", confidence

    pretty_label = label.replace("_", " ").title()
    print(f"✅ Prediction → {pretty_label} ({confidence:.2%} confidence)")
    return label, confidence
# === Test the prediction ===
# if __name__ == "__main__":
#     image_path = "flagged/din.jpg"  
#     label, conf = predict_image(image_path)
