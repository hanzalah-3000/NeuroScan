# NeuroScan

A deep learning project for detecting brain tumors using MRI images.

## 🧠 Overview

NeuroScan uses a convolutional neural network (CNN) to classify and UNet to segment brain MRI scans into tumor vs. non-tumor categories. Built with TensorFlow/Keras, it includes data preprocessing, training, evaluation, and result visualization.

## 🔗 Dataset

Download the classification dataset from this link: [Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

Download the segmenation dataset from this link: [Dataset](https://www.kaggle.com/datasets/atikaakter11/brain-tumor-segmentation-dataset)

Place the extracted dataset folder in the project root directory.

## ⚙️ Requirements

```Python version 3.9.12```
All required Python packages are listed in `requirements.txt`.

## 🚀 Setup

1. Clone this repo:
    ```bash
    git clone https://github.com/hanzalah-3000/NeuroScan.git
    cd NeuroScan
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Download and extract the dataset as described above.

## 💻 Running in Jupyter Notebook

1. To train classification model run the ```ns_classification_model.ipynb``` file
2. To train the segmentation model run the ```ns_highlighting_model.ipynb``` file
3. To view result of existing models run the ```ns_desktop_app.py``` file

## 🔄 Customization

You can substitute model architecture (e.g., CNN, transfer learning like VGG/ResNet) by modifying the notebook. Also, feel free to tweak hyperparameters or data augmentation strategies.
