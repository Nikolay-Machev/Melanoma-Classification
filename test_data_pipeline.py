"""Fast regression tests for metadata labels and grouped splitting."""

import tempfile
import unittest
from pathlib import Path
import pandas as pd

from data_pipeline import lesion_group_split, load_metadata


class DataPipelineTests(unittest.TestCase):
    def setUp(self):
        self.metadata = pd.DataFrame({
            "image_id": [f"image_{i}" for i in range(8)],
            "lesion_id": ["a", "a", "b", "c", "d", "e", "f", "f"],
            "dx": ["mel", "mel", "mel", "mel", "nv", "nv", "nv", "nv"],
        })

    def test_explicit_binary_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.csv"
            self.metadata.to_csv(path, index=False)
            loaded = load_metadata(path)
        self.assertEqual(set(loaded.loc[loaded.dx == "mel", "class_name"]), {"melanoma"})
        self.assertEqual(set(loaded.loc[loaded.dx != "mel", "class_name"]), {"non_melanoma"})

    def test_lesions_never_cross_partitions(self):
        frame = self.metadata.copy()
        frame["class_name"] = frame.dx.eq("mel").map({True: "melanoma", False: "non_melanoma"})
        train, validation = lesion_group_split(frame, validation_size=0.5, seed=4)
        self.assertFalse(set(train.lesion_id).intersection(validation.lesion_id))
        self.assertEqual(len(train) + len(validation), len(frame))


if __name__ == "__main__":
    unittest.main()
