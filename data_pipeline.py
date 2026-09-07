"""Dataset preparation helpers for the HAM10000 binary classification task."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = {"image_id", "lesion_id", "dx"}


def load_metadata(path: str | Path) -> pd.DataFrame:
    metadata = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(metadata.columns)
    if missing:
        raise ValueError(f"metadata is missing columns: {', '.join(sorted(missing))}")
    metadata = metadata.copy()
    metadata["class_name"] = metadata["dx"].eq("mel").map(
        {True: "melanoma", False: "non_melanoma"}
    )
    return metadata


def lesion_group_split(metadata: pd.DataFrame, validation_size=0.2, seed=42):
    """Split by lesion_id so photographs of one lesion stay in one partition."""
    lesions = metadata.groupby("lesion_id", as_index=False)["class_name"].first()
    train_ids, validation_ids = train_test_split(
        lesions["lesion_id"],
        test_size=validation_size,
        random_state=seed,
        stratify=lesions["class_name"],
    )
    train = metadata[metadata["lesion_id"].isin(set(train_ids))].copy()
    validation = metadata[metadata["lesion_id"].isin(set(validation_ids))].copy()
    return train, validation


def index_images(image_directories):
    images = {}
    for directory in image_directories:
        for path in Path(directory).glob("*.jpg"):
            images[path.stem] = path
    return images


def materialize_split(frame, partition, destination, image_index):
    copied = 0
    for row in frame.itertuples(index=False):
        source = image_index.get(row.image_id)
        if source is None:
            raise FileNotFoundError(f"image not found: {row.image_id}.jpg")
        target = Path(destination) / partition / row.class_name / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1
    return copied
