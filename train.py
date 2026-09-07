"""Train a binary EfficientNetB3 melanoma classifier."""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

CLASS_NAMES = ["non_melanoma", "melanoma"]
IMAGE_SIZE = (300, 300)


def make_datasets(data_dir, batch_size, seed):
    options = dict(image_size=IMAGE_SIZE, batch_size=batch_size,
                   class_names=CLASS_NAMES, label_mode="binary")
    train = tf.keras.utils.image_dataset_from_directory(
        Path(data_dir) / "train", shuffle=True, seed=seed, **options
    )
    validation = tf.keras.utils.image_dataset_from_directory(
        Path(data_dir) / "validation", shuffle=False, **options
    )
    autotune = tf.data.AUTOTUNE
    return train.prefetch(autotune), validation.prefetch(autotune)


def build_model():
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.1),
    ], name="training_augmentation")
    backbone = tf.keras.applications.EfficientNetB3(
        include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3)
    )
    backbone.trainable = False
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="melanoma_probability")(x)
    return tf.keras.Model(inputs, outputs), backbone


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="melanoma_precision"),
            tf.keras.metrics.Recall(name="melanoma_recall"),
        ],
    )


def class_weights(dataset):
    labels = np.concatenate([labels.numpy().ravel() for _, labels in dataset])
    counts = np.bincount(labels.astype(int), minlength=2)
    return {index: len(labels) / (2 * count) for index, count in enumerate(counts)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("models/best.keras"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    train, validation = make_datasets(args.data, args.batch_size, args.seed)
    model, backbone = build_model()
    compile_model(model, 1e-3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.output, monitor="val_auc", mode="max", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True),
    ]
    weights = class_weights(train)
    model.fit(train, validation_data=validation, epochs=args.epochs,
              class_weight=weights, callbacks=callbacks)

    backbone.trainable = True
    for layer in backbone.layers[:-40]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    compile_model(model, 1e-5)
    model.fit(train, validation_data=validation, epochs=args.fine_tune_epochs,
              class_weight=weights, callbacks=callbacks)
    print(f"Best model saved to {args.output}")


if __name__ == "__main__":
    main()
