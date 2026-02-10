import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

# ✅ Explicit import (important — avoid keras 3 conflict)
from tensorflow.keras.models import load_model


REPO_ID = "SuYee189/garbage_classification"
MODEL_FILE = "ResNet18_Scratch_best.h5"

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

IMG_SIZE = (224, 224)


def load_model_from_hf():

    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILE,
        local_dir="models"
    )

    # ✅ compile=False avoids keras version conflict
    model = load_model(model_path, compile=False)

    return model


# ======================
# PREPROCESS
# ======================

def preprocess_image(img):

    img = img.resize(IMG_SIZE)
    img_array = np.array(img)

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# ======================
# PREDICT
# ======================

def predict_image(img, model):

    img_array = preprocess_image(img)

    preds = model.predict(img_array)

    class_index = np.argmax(preds)
    confidence = float(np.max(preds))

    return CLASS_NAMES[class_index], confidence
