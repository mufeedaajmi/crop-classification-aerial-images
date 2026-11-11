import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

IMAGE_SIZE = (224, 224)
MODEL_PATH = Path("models/crop_classifier.keras")
CLASS_NAMES_PATH = Path("models/class_names.json")


def predict(image_path):
    if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            "Model files not found. Train the model first using: python src/train.py"
        )

    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = json.loads(CLASS_NAMES_PATH.read_text())

    image = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    probabilities = model.predict(image_array, verbose=0)[0]
    best_index = int(np.argmax(probabilities))

    print(f"Predicted crop: {class_names[best_index]}")
    print(f"Model confidence: {probabilities[best_index] * 100:.2f}%")

    print("\nTop predictions:")
    for index in np.argsort(probabilities)[::-1][:3]:
        print(f"- {class_names[int(index)]}: {probabilities[index] * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a crop image.")
    parser.add_argument("image", help="Path to an aerial crop image.")
    args = parser.parse_args()
    predict(args.image)
