# Deep Learning vision models architectures from scratch

## Overview
This repository contains a collection of machine learning practical projects completed during a Deep Learning course.  
The goal of these projects was to **implement neural network architectures from scratch** to understand their internal mechanics rather than relying on high-level frameworks.
It features implementations of neural networks, convolutional models, VGG-like architectures, and U-Net for image segmentation.

> *Note: The focus of this repository is on understanding model internals.
Most architectures are implemented from scratch without using high-level APIs.*

## Details

As indicated above, several kind of models were implemented, simple ones like **Fully Connected NN** and relatively small **CNN**. There were implemented with *TensorFlow* and trained on different datasets like *MNIST*, *CIFAR10* or the [*Iris*](https://www.kaggle.com/datasets/uciml/iris) dataset.
**U-Net** and **VGG16** models were implemented in *PyTorch* regarding of the following architectures and respectively trained on *CIFAR100* and [*CamVid*](https://www.kaggle.com/datasets/carlolepelaars/camvid?resource=download) datasets.


UNet            |  VGG16
:-------------------------:|:-------------------------:
![UNet Architecture](images/unet.webp) | ![VGG16 Architecture](images/VGG-16.png)


## Structure

```
.
├── data/
├── images/                 # illustration and output images
├── notebooks/              # fcnn and cnn tensorflow implementation
├── unet/                   # pytorch unet implementation
├── vgg/                    # pytorch vgg implementation
├── README.md
├── exp_unet.ipynb          # experimentation notebooks
├── exp_vgg.ipynb           # experimentation notebooks
└── utils.py                # plot function
```

## How to run

1. Clone the repository:
```
git clone https://github.com/yourusername/deep-learning-architectures-from-scratch.git
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run a model:
```
python vgg/train.py
```

## Learning Objectives
These implementations were developed for educational purposes as part of a university course, in order to explore CNN architectures and learn encoder-decoder and skip connections.

## Results
Img Initial            |  Mask
:-------------------------:|:-------------------------:
![UNet Initial picture](images/img_initial.png) | ![U-Net Segmentation Example](images/mask.png)

UNet Performance            |  VGG Performance
:-------------------------:|:-------------------------:
![U-Net Performance]() | ![VGG Performance]()



## Author
Project created by Antony Manuel, as part of the FDAA course, under Pr [Sebastien Ambellouis](github.com/SebAmb) supervision.

IMT Nord Europe — 2025–2026