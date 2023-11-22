"""
This module defines a Trainer class for training a neural network model on a given dataset.
It includes functionalities for training the model, computing loss and accuracy, performing 
backward propagation, and saving historical data for analysis. The training process involves
both a training and a validation phase, with metrics logged for each.
"""

import logging
import argparse
import torch
import sys
import os
import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Defining path constants for model and data
PATH = "/Users/shahmuhammadraditrahman/Desktop/IrisClassifier/iris_classifier"

sys.path.append(PATH)

import config

# Setting up logging configuration
logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(config.LOGS_PATH, "train_model.log"),
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


# Importing necessary utility functions and the Classifier model
from utils.utils import (
    define_loss_function as loss_function,
    define_optimizer as optimizer,
    load_pickle,
    create_pickle,
)
from models.model import Classifier
from visualization.visualize import Visualizer


class Trainer:
    """
    The Trainer class encapsulates the functionality for training a machine learning model.
    It includes methods for handling the training process, computing loss and accuracy,
    backpropagation, and storing training history.
    """

    def __init__(self, epochs=200, lr=0.01, display=True):
        """
        Initializes the Trainer with specified epochs, learning rate, and display setting.
        Loads the training and testing datasets, initializes the model, loss function, and optimizer.
        Sets up a history dictionary to store training and validation loss and accuracy.
        """
        logging.info("Initialization is processing.".capitalize())

        self.epochs = epochs
        self.learning_rate = lr
        self.display = display
        self.train_loader = load_pickle(
            os.path.join(config.DATA_PATH, "train_loader.pkl")
        )
        self.test_loader = load_pickle(
            os.path.join(config.DATA_PATH, "test_loader.pkl")
        )
        self.model = Classifier()

        self.loss_function = loss_function()
        self.optimizer = optimizer(model=self.model, lr=self.learning_rate)
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "test_loss": [],
            "test_accuracy": [],
        }

    def convert_to_long(self, label):
        """
        Converts label data to long tensor format, which is required for certain types of loss functions in PyTorch.
        """
        logging.info("Converting labels to long tensor format.".capitalize())

        return torch.Tensor(label).long()

    def _predict_and_evaluate_loss(self, dataset, specify, epoch):
        """
        Makes predictions on the given dataset and evaluates the loss.
        It also records the actual and predicted labels for accuracy computation.
        The 'specify' parameter determines whether to perform backpropagation ('train') or not ('test').
        """
        logging.info("Making predictions on the given dataset.".capitalize())

        actual = []
        predict = []
        loss_compute = []
        for data, label in dataset:
            label = self.convert_to_long(label=label)

            prediction = self.model(data)
            loss = self._compute_loss(prediction=prediction, label=label)

            if specify != "test":
                self._do_backward_propagation(loss=loss)
                torch.save(
                    self.model,
                    os.path.join(
                        config.MODEL_CHECK_POINT_PATH, f"model_{epoch + 1}.pth"
                    ),
                )

        actual.extend(label)
        predict.extend(torch.argmax(prediction, dim=1))
        loss_compute.append(loss.item())

        return actual, predict, np.array(loss_compute).mean()

    def _do_backward_propagation(self, loss):
        """
        Performs the backpropagation algorithm: resetting gradients, computing gradient, and updating model parameters.
        """
        logging.info("Performing backpropagation.".capitalize())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _compute_loss(self, prediction, label):
        """
        Computes the loss between the predicted and actual labels using the defined loss function.
        """
        logging.info("Computing loss.".capitalize())

        loss = self.loss_function(prediction, label)
        return loss

    def _compute_accuracy(self, actual, predict):
        """
        Computes the accuracy of predictions by comparing with actual labels.
        """
        logging.info("Computing accuracy.".capitalize())

        return accuracy_score(predict, actual)

    def _save_historical_data(self, loss, accuracy, specify):
        """
        Saves the computed loss and accuracy in the training history for later analysis.
        """
        logging.info("Saving historical data.".capitalize())

        self.history["{}_loss".format(specify)].append(loss)
        self.history["{}_accuracy".format(specify)].append(accuracy)

    def _display(self, **data):
        """
        Displays the training progress and metrics if the display option is enabled.
        """
        logging.info("Displaying progress.".capitalize())

        print("Epochs - {}/{}".format(data["epoch"], data["total_epochs"]))
        print(
            "[===========] train_loss:{} - train_accuracy:{} - val_loss:{} - val_accuracy:{}".format(
                data["train_loss"],
                data["train_accuracy"],
                data["val_loss"],
                data["val_accuracy"],
            )
        )

    def train(self):
        """
        Executes the training process for the defined number of epochs.
        It involves training the model on the training dataset and validating it on the test dataset.
        """
        logging.info("Starting training process.".capitalize())

        for epoch in range(self.epochs):
            (
                train_actual,
                train_predict,
                train_total_loss,
            ) = self._predict_and_evaluate_loss(
                dataset=self.train_loader, specify="train", epoch=epoch
            )

            (val_actual, val_predict, val_total_loss) = self._predict_and_evaluate_loss(
                dataset=self.test_loader, specify="test", epoch=epoch
            )

            logging.info("Epoch: {}/{}".format(epoch + 1, self.epochs))

            train_accuracy = self._compute_accuracy(
                actual=train_actual, predict=train_predict
            )
            val_accuracy = self._compute_accuracy(
                actual=val_actual, predict=val_predict
            )

            logging.info("Saving the loss and train in the history".title())
            self._save_historical_data(
                loss=train_total_loss, accuracy=train_accuracy, specify="train"
            )
            self._save_historical_data(
                loss=val_total_loss, accuracy=val_accuracy, specify="test"
            )

            if self.display:
                self._display(
                    epoch=epoch,
                    total_epochs=self.epochs,
                    train_loss=train_total_loss,
                    train_accuracy=train_accuracy,
                    val_loss=val_total_loss,
                    val_accuracy=val_accuracy,
                )
            else:
                logging.info("Nothing is showing.".title())

    def model_performance(self):
        """
        Visualizes the training history of a machine learning model.

        Loads training history from a pickle file and displays loss and accuracy metrics
        using the Visualizer class. Assumes history contains 'train_loss', 'test_loss',
        'train_accuracy', and 'test_accuracy'.
        """
        logging.info("Visualizing model performance".title())

        history = load_pickle(filename=os.path.join(config.MODEL_PATH, "history.pkl"))

        Visualizer().show_model_performance(
            train_loss=history["train_loss"], val_loss=history["test_loss"]
        )
        Visualizer().show_model_performance(
            train_accuracy=history["train_accuracy"],
            val_accuracy=history["test_accuracy"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs", type=int, default=1000, help="number of Epochs".title()
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate".title())
    parser.add_argument("--display", type=bool, default=True, help="display".title())

    args = parser.parse_args()

    if args.epochs and args.lr and args.display:
        model_trainer = Trainer(args.epochs, args.lr, args.display)
        model_trainer.train()
        model_trainer.model_performance()

    else:
        logging.exception("Define is not correct".title())
