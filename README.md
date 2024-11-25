# This repository is a reimplementation of EfficientViT

This repository provides a reimplementation of **EfficientViT**, a state-of-the-art vision transformer model designed for high performance and efficiency in image classification tasks.

## Overview

EfficientViT is an efficient Vision Transformer (ViT) model designed for scalable and faster training on large datasets like ImageNet. This implementation follows the original architecture and training strategy, with a few optimizations for ease of use and better performance. The repository consists of two main components:

1. **`data_downloader.py`**: A script to download the ImageNet-100 dataset from Hugging Face.
2. **`dist_train.py`**: A script to train the model from scratch using the downloaded dataset.

## Requirements

Before you start, ensure you have the following dependencies installed. You can install them directly from the `requirements.txt` file.

### Install Dependencies

To set up the required environment, run the following command:

```bash
pip install -r requirements.txt
