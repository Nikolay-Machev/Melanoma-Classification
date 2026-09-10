# Melanoma Classifier

[![Tests](https://github.com/Nikolay-Machev/Melanoma-Classification/actions/workflows/tests.yml/badge.svg)](https://github.com/Nikolay-Machev/Melanoma-Classification/actions/workflows/tests.yml)

I built a binary skin-lesion classifier with EfficientNetB3 and transfer learning to distinguish melanoma from other lesion classes in the HAM10000 dataset.

**Research and educational use only — this project is not a medical device and must not be used for diagnosis.**

**[HAM10000 dataset on Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)**

## How it works

1. **Data preparation** — I map the original seven diagnostic categories into `melanoma` and `non_melanoma`. I split the data by `lesion_id`, not by individual image, so multiple photographs of the same lesion cannot appear in both training and validation.
2. **Model** — I use an ImageNet-pretrained EfficientNetB3 backbone with training-only augmentation, global average pooling, dropout, and a sigmoid output representing melanoma probability. The pipeline first trains the classification head and then fine-tunes the final backbone layers at a lower learning rate.
3. **Evaluation and inference** — I report accuracy, ROC AUC, melanoma precision, and melanoma recall with melanoma explicitly encoded as the positive class. `predict.py` runs a single dermoscopic image through a saved model.

## What this project demonstrates

- Building a reproducible transfer-learning and inference pipeline
- Preventing patient-proxy leakage through lesion-grouped validation
- Separating exploratory results from evidence that would support clinical claims

## Original experiment

My original notebook trained on all **10,015 HAM10000 images**: 1,113 melanoma and 8,902 non-melanoma images.

These results document the original experiment; they are not results from the repository's newer leakage-resistant evaluation pipeline.

| Evaluation | Result | Interpretation |
|---|---:|---|
| Original image-level validation split | 85.26% peak validation accuracy | Preliminary notebook result; repeated-lesion leakage was not controlled |
| Original image-level validation split | 0.888 peak validation AUC | Preliminary result from a different epoch; not an external or lesion-held-out estimate |

I treat these as preliminary results. The notebook used an image-level split, which may place different photographs of one lesion in both partitions, and its folder ordering made the reported precision and recall describe the non-melanoma class. The reproducible scripts in this repository correct both issues; their results should be reported separately after retraining.

## Project structure

```text
├── data_pipeline.py         # Metadata validation and lesion-grouped splitting
├── prepare_data.py          # Build train/validation image directories
├── train.py                 # Train and fine-tune EfficientNetB3
├── predict.py               # Run inference on one dermoscopic image
├── test_data_pipeline.py    # Regression tests for labels and split integrity
├── Melanoma_Model.ipynb     # Original exploratory experiment and results
├── requirements.txt         # Python dependencies
├── .gitignore               # Excludes datasets, environments, and model weights
└── README.md
```

## Setup

```bash
git clone https://github.com/Nikolay-Machev/Melanoma-Classification.git
cd Melanoma-Classification
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download `HAM10000_metadata.csv` and both HAM10000 image archives from the dataset page, then extract the image archives.

## Usage

**Prepare leakage-resistant partitions:**

```bash
python prepare_data.py \
  --metadata /path/to/HAM10000_metadata.csv \
  --images /path/to/HAM10000_images_part_1 /path/to/HAM10000_images_part_2
```

This creates the following ignored local structure:

```text
data/
├── train/
│   ├── melanoma/
│   └── non_melanoma/
└── validation/
    ├── melanoma/
    └── non_melanoma/
```

**Train and fine-tune the model:**

```bash
python train.py
```

The best validation-AUC checkpoint is saved to `models/best.keras`.

**Classify one image:**

```bash
python predict.py /path/to/dermoscopic_image.jpg
```

## Testing

```bash
python -m unittest -v
```

The tests verify the melanoma label mapping and ensure that a `lesion_id` can never cross the training/validation boundary.

## Methodological notes

- I apply augmentation only during training; validation images remain unchanged.
- I keep the pretrained EfficientNet input in its expected pixel range instead of rescaling it twice.
- I freeze batch-normalization layers during fine-tuning to protect their pretrained statistics.
- Because HAM10000 is imbalanced, I use class weighting and report discrimination and class-specific metrics alongside accuracy.
- A robust clinical claim would require an untouched external test set, threshold selection, calibration analysis, subgroup analysis, and prospective validation. This repository does not provide those.

## License

This project is licensed under the MIT License—see the [LICENSE](LICENSE) file for details.

## Machine-learning portfolio

Part of my machine-learning portfolio, spanning [models built from scratch](https://github.com/Nikolay-Machev/Tic-tac-toe-AI-Bot), computer vision, and [scientific machine learning](https://github.com/Nikolay-Machev/Qsar-Solubility-Predictor).
