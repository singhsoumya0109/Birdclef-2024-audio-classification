# BirdCLEF 2024 Audio Classification

This repository contains the implementation for the BirdCLEF 2024 competition on Kaggle. The goal of this project is to build a Convolutional Neural Network (CNN) model to classify bird species from audio recordings.

## Table of Contents

- [Dataset](#dataset)
- [Feature Extraction](#feature-extraction)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Prediction](#prediction)
- [Submission](#submission)
- [Requirements](#requirements)
- [Acknowledgements](#acknowledgements)
- [Contributors](#contributors)

## Dataset

The dataset used in this competition consists of audio recordings of various bird species. The dataset and metadata are provided by Kaggle and can be found [here](https://www.kaggle.com/competitions/birdclef-2024/data).

## Feature Extraction

We use mel-spectrograms to represent the audio data. Mel-spectrograms are a type of time-frequency representation that are well-suited for audio classification tasks. The `librosa` library is used for this purpose.

## Model Architecture

The model is a Convolutional Neural Network (CNN) built using TensorFlow and Keras. The architecture includes:

- Three convolutional layers with batch normalization and max pooling
- A fully connected (dense) layer with dropout for regularization
- An output layer with softmax activation for classification

## Training the Model

The model is trained using the following callbacks:

- `EarlyStopping` to prevent overfitting
- `ReduceLROnPlateau` to adjust the learning rate when the validation loss plateaus
- A custom `MinimumEpochs` callback to ensure a minimum number of epochs are completed

## Prediction

For prediction, we process the test audio files in segments, extract features, and use the trained model to classify the bird species.

## Submission

The predictions are saved in a CSV file in the format required by Kaggle for submission.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- Librosa
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Tqdm

## Acknowledgements

This project is part of the BirdCLEF 2024 competition on Kaggle. We thank Kaggle and the dataset providers for making this competition possible.

## Contributors

- [Himadri Rajsekhar Giri](https://github.com/Himadri-1801)
- [Suman Khara](https://github.com/Suman-Khara)
- [Debatra Banerjee](https://github.com/debatra2004)
