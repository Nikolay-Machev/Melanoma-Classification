"""Create leakage-resistant train/validation directories from HAM10000."""

import argparse
from pathlib import Path

from data_pipeline import index_images, lesion_group_split, load_metadata, materialize_split


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True,
                        help="one or both HAM10000 image directories")
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    metadata = load_metadata(args.metadata)
    train, validation = lesion_group_split(metadata, args.validation_size, args.seed)
    image_index = index_images(args.images)
    train_count = materialize_split(train, "train", args.output, image_index)
    validation_count = materialize_split(validation, "validation", args.output, image_index)
    overlap = set(train.lesion_id).intersection(validation.lesion_id)
    if overlap:
        raise RuntimeError("lesion leakage detected between partitions")
    print(f"Prepared {train_count:,} training and {validation_count:,} validation images")
    print(f"Unique lesions: {train.lesion_id.nunique():,} train / {validation.lesion_id.nunique():,} validation")


if __name__ == "__main__":
    main()
