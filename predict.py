"""Run one dermoscopic image through a trained model."""

import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/best.keras"))
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    image = tf.keras.utils.load_img(args.image, target_size=(300, 300))
    batch = np.expand_dims(tf.keras.utils.img_to_array(image), axis=0)
    probability = float(tf.keras.models.load_model(args.model).predict(batch, verbose=0)[0, 0])
    label = "melanoma" if probability >= args.threshold else "non-melanoma"
    print(f"Prediction: {label}")
    print(f"Melanoma probability: {probability:.2%}")
    print("Research demonstration only — not a medical diagnosis.")


if __name__ == "__main__":
    main()
