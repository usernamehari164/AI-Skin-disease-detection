import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import sys

# ✅ Get current file directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Absolute paths (FIX)
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_final.keras")

# ✅ Load model
model = tf.keras.models.load_model(MODEL_PATH)

# ✅ HARDCODED ORIGINAL MAPPING (To prevent IndexErrors)
# This mapping matches the 20 output neurons of your model_final.keras
FULL_CLASS_MAP = {
    0: "Acne Vulgaris", 1: "Actinic Keratoses", 2: "Basal Cell Carcinoma", 
    3: "Benign Keratosis", 4: "Cellulitis", 5: "Dermatofibroma", 
    6: "Eczema", 7: "Folliculitis", 8: "Impetigo", 9: "Melanoma", 
    10: "Nevus (Mole)", 11: "Psoriasis", 12: "Ringworm (Tinea)", 
    13: "Rosacea", 14: "Scabies", 15: "Seborrheic Dermatitis", 
    16: "Urticaria (Hives)", 17: "Vascular Lesions", 18: "Vitiligo", 
    19: "Warts (HPV)"
}

# ✅ USER-ALLOWED DISEASES (The 10 you want to support)
ALLOWED_DISEASES = [
    "Acne Vulgaris", "Actinic Keratoses", "Basal Cell Carcinoma", 
    "Eczema", "Folliculitis", "Melanoma", "Psoriasis", 
    "Ringworm (Tinea)", "Vitiligo", "Warts (HPV)"
]

CONFIDENCE_THRESHOLD = 0.60

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image_path):
    img = preprocess_image(image_path)
    preds = model.predict(img, verbose=0)
    preds = preds[0] 

    idx = int(np.argmax(preds))
    confidence = float(preds[idx])

    # Get the name from our internal 20-class map
    raw_label = FULL_CLASS_MAP.get(idx, "Uncertain")

    # Final Filter: Must be in the allowed list AND meet confidence
    if confidence >= CONFIDENCE_THRESHOLD and raw_label in ALLOWED_DISEASES:
        label = raw_label
    else:
        label = "Uncertain"

    return {
        "prediction": label,
        "confidence": round(confidence * 100, 2)
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = predict(image_path)
        print(json.dumps(result))