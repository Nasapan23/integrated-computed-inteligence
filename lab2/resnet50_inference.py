from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Image classification using pretrained ResNet50 (ImageNet)."
    )
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Path to a local image file (jpg/png/etc.).",
    )
    parser.add_argument(
        "--top-k",
        default=5,
        type=int,
        help="How many top classes to print (default: 5).",
    )
    return parser.parse_args()


def load_image(img_path: Path) -> np.ndarray:
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    model = ResNet50(weights="imagenet")
    processed = load_image(args.image)
    predictions = model.predict(processed, verbose=0)
    decoded = decode_predictions(predictions, top=args.top_k)[0]

    print(f"Image: {args.image.resolve()}")
    print(f"Top {args.top_k} predictions:")
    for idx, (_, label, probability) in enumerate(decoded, start=1):
        print(f"{idx}. {label:<20} {probability * 100:6.2f}%")


if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    main()
