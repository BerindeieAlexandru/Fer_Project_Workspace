# ResEmoteNet: Bridging Accuracy and Loss Reduction in Facial Emotion Recognition

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/resemotenet-bridging-accuracy-and-loss/facial-expression-recognition-on-affectnet)](https://paperswithcode.com/sota/facial-expression-recognition-on-affectnet?p=resemotenet-bridging-accuracy-and-loss)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/resemotenet-bridging-accuracy-and-loss/facial-expression-recognition-on-fer2013)](https://paperswithcode.com/sota/facial-expression-recognition-on-fer2013?p=resemotenet-bridging-accuracy-and-loss)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/resemotenet-bridging-accuracy-and-loss/facial-expression-recognition-on-raf-db)](https://paperswithcode.com/sota/facial-expression-recognition-on-raf-db?p=resemotenet-bridging-accuracy-and-loss)

A new network that helps in extracting facial features and predict the emotion labels.

The emotion labels in this project are:

- Happiness 😀
- Surprise 😦
- Anger 😠
- Sadness ☹️
- Disgust 🤢
- Fear 😨
- Neutral 😐

## Table of Content

- [Installation](#installation)
- [Usage](#usage)
- [Checkpoints](#checkpoints)
- [Results](#results)
- [License](#license)

## Installation

1. Create a Conda environment.

```bash
conda create --n "fer"
conda activate fer
```

2. Install Python v3.8 using Conda.

```bash
conda install python=3.8
```

3. Clone the repository.

```bash
git clone https://github.com/ArnabKumarRoy02/ResEmoteNet.git
```

4. Install the required libraries.

```bash
pip install -r requirement.txt
```

## Usage

Run the file.

```bash

cd train_files
python ResEmoteNet_train.py
```

## Checkpoints

All of the checkpoint models for FER2013, RAF-DB and AffectNet-7 can be found [here](https://drive.google.com/drive/folders/1Daxa6d1-XFxxpg6dyxYl4V-anfiHwtqK?usp=sharing).

## Results

- FER2013:
  - Testing Accuracy: **79.79%** (SoTA - 76.82%)
- CK+:
  - Testing Accuracy: **100%** (SoTA - 100%)
- RAF-DB:
  - Testing Accuracy: **94.76%** (SoTA - 92.57%)
- FERPlus:
  - Testing Accuracy: 91.64% (SoTA - **95.55%**)
- AffectNet (7 emotions):
  - Testing Accuracy: **72.93%** (SoTA - 69.4%)

## Extra experiments

For this method, we first tried to replicate the results provided by the authors. This can be seen in [fer_2013_original_train](./fer2013_original_train/) where we trained the model proposed by the authors on fer2013 dataset and evaluated it, also saving best checkpoints in [snapshots](./fer2013_original_train/snapshots/).

We then tried in [fer_2013_balanced_train](./fer2013_balanced_train/) to apply same approach on the custom created dataset, that is a balanced version of fer2013 that has exactly the same amount of sample in all classes. We evaluated it and also saved the best checkpoints in [snapshots](./fer2013_balanced_train/snapshots/)

In [fer+_zone](./fer+_zone/) we conducted our experiments on Fer+ dataset to check authors results and see how the model performs, also saved the best checkpoints.

In [ensemble](./ensemble/) we conducted experiments using original dataset, where we tried to change some hyperparamters and to build an snapshot ensemble based on model best performing checkpoints. We also tried to fine tune the model on different variations of dataset splits to see if we could increase its performance.

In [results_comparison](./results_comparison/) we will centralize the results obtained during different experiments in order to be able to compare how models perform and how dataset changes affects the performance.

## License

This repository is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.
