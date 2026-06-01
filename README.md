# Road Segmentation from Satellite Imagery

A deep learning model for automatic extraction and segmentation of road networks from satellite/aerial imagery using semantic segmentation.

---

## Overview

This project automates the detection and extraction of road networks from satellite imagery using a U-Net architecture with EfficientNetB4 backbone. The model handles the challenging task of identifying thin, linear road features while managing severe class imbalance (roads typically occupy only ~5% of image pixels).

### Key Features
- Custom weighted Dice loss for class imbalance handling
- Advanced augmentation pipeline optimized for satellite data
- Pre-trained EfficientNetB4 backbone for better feature extraction
- Automatic learning rate scheduling and model checkpointing
- Efficient TensorFlow data pipeline with prefetching
- Multi-scale attention
- Comprehensive evaluation metrics (IoU, Dice, loss tracking)

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | TensorFlow / Keras |
| **Architecture** | U-Net + EfficientNetB4 |
| **Loss Function** | Weighted Dice Loss + BCE |
| **Data Augmentation** | Albumentations |
| **Image Processing** | OpenCV, NumPy Pandas |
| **Metric** | Mean Intersection over Union (mIoU) |
| **Language** | Python 3.10 |

---

## Installation

### Prerequisites
```bash
Python >= 3.8
CUDA 11.0+ (optional, for GPU acceleration)
```

### Clone Repository
```bash
git clone https://github.com/MarwanBrt/Road-Extraction-from-satellite-Images/
cd Road-Extraction-from-satellite-Images
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### requirements.txt
```
tensorflow>=2.10.0
keras>=2.10.0
albumentations>=1.3.0
opencv-python>=4.7.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
```

---

## Dataset Structure

```
## Download data
!curl -L -o deepglobe-road-extraction-dataset.zip https://www.kaggle.com/api/v1/datasets/download/balraj98/deepglobe-road-extraction-dataset
## Create folders
data_path =./data"
model_path = "./models"
os.makedirs(data_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)
## Unzip data to folders
!unzip ./deepglobe-road-extraction-dataset.zip -d data
```


## Data Augmentation Pipeline

The project uses advanced augmentation optimized for satellite imagery:

```python
A.Compose([
  A.Resize(img_size, img_size),
  A.RandomRotate90(p=0.4),
  A.VerticalFlip(p=0.4),
  A.HorizontalFlip(p=0.3),
  A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.25),
  A.CoarseDropout(max_holes=6,max_height=img_size//12,max_width=img_size//12, p=0.4),
  A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.25),
  A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15,p=0.3),
  A.CLAHE(p=1, clip_limit=8),
  A.Normalize(mean=mean, std=std),
])
```


## Project Structure

```
Road-Extraction-from-satellite-Images/
├── data/
│   ├── test/
│   └── train/
│   └── valid/
│   └── class_dict.csv/
│   └── metadata.csv/
├── models/
│   └── best_model.h5
├── README.md
├── requirements.txt
└── .gitignore
```

---

## References

- **Dataset:** [DeepGlobe Road Extraction Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset)
- **U-Net:** [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- **EfficientNet:** [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
- **Dice Loss:** [Milletari et al., 2016](https://arxiv.org/abs/1606.04797)
- **Albumentations:** [https://albumentations.ai/](https://albumentations.ai/)

---

## Author

**Marouane Barati**
- GitHub: [@Escobar-12](https://github.com/Escobar-12)
- Email: marwanbrt12@gmail.com

---

**Last Updated:** April 2026  
