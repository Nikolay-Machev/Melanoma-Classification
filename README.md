# Melanoma-Classification
Deep learning model classifying skin lesions as melanoma vs. non-melanoma using HAM10000 images. Built with EfficientNetB3, achieving ~85% accuracy and AUC ~0.88 for AI-assisted diagnosis.

## Project Overview

This project implements a convolutional neural network to classify dermatoscopic images into melanoma and non-melanoma classes using the HAM10000 dataset. Leveraging EfficientNetB3 and transfer learning, it achieves high accuracy, demonstrating the potential for AI-assisted skin cancer diagnosis.

## Model Performance

- Validation Accuracy: ~85%
- Validation AUC: ~0.88
- Precision: up to 98%
- Recall: up to 87%

## Dataset

This project uses the HAM10000 dataset:
[HAM10000 on Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)

## Download the Trained Model

The trained EfficientNetB3 model (~135 MB) is available here:

[Download efficientnet_model_finetuned.h5](https://drive.google.com/file/d/16Z1aQTZVtOJg5HPoD37LwSxyqh9dsVFL/view?usp=sharing)

After downloading, place the file in your project folder and load it in your notebook like this:

```python
from tensorflow.keras.models import load_model

model = load_model('efficientnet_model_finetuned.h5')
