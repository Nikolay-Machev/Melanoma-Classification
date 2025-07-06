# Melanoma-Classification
Deep learning model classifying skin lesions as melanoma vs. non-melanoma using HAM10000 images. Built with EfficientNetB3, achieving ~85% accuracy and AUC ~0.88 for AI-assisted diagnosis.
## Download the Trained Model

The trained EfficientNetB3 model (~135 MB) is available here:

[Download efficientnet_model_finetuned.h5](https://drive.google.com/file/d/16Z1aQTZVtOJg5HPoD37LwSxyqh9dsVFL/view?usp=sharing)

After downloading, place the file in your project folder and load it in your notebook like this:

```python
from tensorflow.keras.models import load_model

model = load_model('efficientnet_model_finetuned.h5')
