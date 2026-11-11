# Crop Classification Using Aerial Images

A Python image-classification starter project that uses transfer learning with
MobileNetV2 to classify crop images into the categories supplied in the dataset.

> Important: This repository contains the training and prediction code. You must
> add a suitable, correctly labelled aerial-image dataset and train the model
> before making predictions. Do not report accuracy until you have evaluated it
> on a separate test set.

## 1. Project structure

Arrange your dataset like this. Replace the example class names with the crop
categories present in your dataset.

```text
crop-classification-aerial-images/
├── data/
│   ├── train/
│   │   ├── crop_class_1/
│   │   └── crop_class_2/
│   └── validation/
│       ├── crop_class_1/
│       └── crop_class_2/
├── models/                 # created during training; not committed to Git
├── src/
│   ├── train.py
│   └── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

Each class folder should contain images belonging to that class. Training and
validation folders must have the same class subfolder names. Keep a separate
test set aside for final evaluation; do not use test images during training.

## 2. Set up

Use Python 3.10 or 3.11 in a virtual environment.

### Windows

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

TensorFlow installation support depends on your operating system and hardware.
If installation fails, check the official TensorFlow installation guide.

## 3. Train

After arranging the images in `data/train` and `data/validation`, run:

```bash
python src/train.py
```

The script uses MobileNetV2 pretrained on ImageNet as a frozen feature extractor,
adds a classification head, and saves the trained model and class names in
`models/`.

## 4. Predict

After training, run:

```bash
python src/predict.py path/to/your/aerial_image.jpg
```

The script prints the predicted class, confidence, and up to three highest
scoring classes.

## 5. Notes and limitations

- This is a baseline implementation, not a claim of measured performance.
- The example pipeline resizes images to 224 x 224 pixels. Very small crop
  patterns or high-resolution remote-sensing details may be lost.
- The model predicts only the classes represented in your labelled dataset.
- Validate on geographically separate or otherwise independent test imagery
  where possible to reduce data leakage.
- Review class balance, image quality, labels, and confusion between similar
  crop types before relying on predictions.
- Large datasets and trained model files are excluded from Git by `.gitignore`.
  Keep a note in your project documentation describing the dataset source and
  its licence.

## 6. Suggested future improvements

- Evaluate on a held-out test set and report accuracy, precision, recall,
  F1-score, and a confusion matrix.
- Fine-tune selected MobileNetV2 layers after training the classification head.
- Compare with a model suited to multispectral imagery if your aerial dataset
  contains bands beyond ordinary RGB.
