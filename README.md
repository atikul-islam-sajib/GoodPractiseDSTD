# Iris Classifier

## Introduction

The Iris Classifier is a Python-based machine learning application designed to classify Iris species based on given features. This project demonstrates the power of machine learning algorithms in classifying biological species.

## Features

- Data preprocessing
- Model training with customizable hyperparameters
- Predictive functionality on new data

## Prerequisites

- Python 3.x
- Required libraries: [list any libraries your project depends on]

## Installation

```bash
Clone the repository to your local machine:

git clone [repository URL]

cd /IrisClassifier/iris_classifier

```

## Usage

The application can be used in several ways:

### 1. Data Preprocessing and Model Training

To preprocess your data and train the model:

```bash
python iris_classifier/command_line.py --dataset /path/to/IRIS.csv --split 0.20 --preprocessing

```

- `--dataset`: Path to the dataset
- `--split`: Fraction of data to be used as the test set

### 2. Custom Training Parameters

To train the model with custom epochs and learning rate, and to display training progress:

```bash
python iris_classifier/command_line.py --epochs 200 --lr 0.001 --display True

```

- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--display`: Set to True to display training progress

### 3. Making Predictions

To make predictions on new data:
python iris_classifier/command_line.py --predict

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any queries, please contact [Atikul Islam Sajib] at [atikul.sajib@ptb.de].
