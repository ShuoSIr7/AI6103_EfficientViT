# This repository is a reimplementation of EfficientViT

This repository provides a reimplementation of EfficientViT-M0, one of the high-speed vision transformers famlies designed to address the computational challenges of traditional Vision Transformers. 

## Overview

To train an **EfficicentViT-M0** on **ImageNet100** dataset and reproduce the results, you can simply run:

1. **`data_downloader.py`**: A script to download the ImageNet-100 dataset from Hugging Face.
2. **`dist_train.py`**: A script to train the model from scratch using the downloaded dataset.

## Requirements

Before you start, ensure you have the following dependencies installed. You can install them directly from the `requirements.txt` file.

### Install Dependencies

To set up the required environment, run the following command:

```bash
pip install -r requirements.txt
