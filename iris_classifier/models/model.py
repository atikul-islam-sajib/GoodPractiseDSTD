""" This script defines a PyTorch neural network model, Classifier, 
    with specific layers and configurations. It also sets up logging 
    and command-line argument parsing for model configuration.
"""

import torch
import torch.nn as nn
import logging
import argparse
from collections import OrderedDict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename="/Users/shahmuhammadraditrahman/Desktop/IrisClassifier/logs/model.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

import sys

sys.path.append("/Users/shahmuhammadraditrahman/Desktop/IrisClassifier/iris_classifier")

from utils.utils import create_pickle


class Classifier(nn.Module):
    """Defines a Classifier neural network with separate left, middle,
    right pathways and a fully connected layer.
    """

    def __init__(self):
        """Initializes the Classifier model with specific layer configurations."""
        super(Classifier, self).__init__()

        # Define three pathways with different configurations
        self.left = self._make_layers(layers_config=[(4, 32), (32, 16)], prefix="left")
        self.middle = self._make_layers(
            layers_config=[(4, 64), (64, 32), (32, 16)], prefix="middle"
        )
        self.right = self._make_layers(
            layers_config=[(4, 16), (16, 32), (32, 16)], prefix="right"
        )

        # Define a fully connected layer
        self.fc = self._connected_layer(
            layers_config=[(16 + 16 + 16, 32, 0.3), (32, 16, 0.7), (16, 3)], prefix="fc"
        )

    def _make_layers(self, layers_config: list, prefix: str):
        """Creates a sequence of layers based on the given configuration."""
        layers = OrderedDict()
        for idx, (input_features, output_features) in enumerate(layers_config):
            layers["{}-{}".format(prefix, idx)] = nn.Linear(
                in_features=input_features, out_features=output_features
            )
            layers[f"{prefix}-ReLU"] = nn.ReLU()

        return nn.Sequential(layers)

    def _connected_layer(self, layers_config: list, prefix: str):
        """Creates a fully connected layer with dropout based on the given configuration."""
        layers = OrderedDict()
        for idx, (input_features, output_features, dropout) in enumerate(
            layers_config[:-1]
        ):
            layers["{}-{}".format(prefix, idx)] = nn.Linear(
                in_features=input_features, out_features=output_features
            )
            layers[f"{prefix}-ReLU"] = nn.ReLU()
            layers[f"{prefix}-Dropout"] = nn.Dropout(p=dropout)

        # Last layer without dropout
        (input_features, output_features) = layers_config[-1]
        layers["out"] = nn.Linear(
            in_features=input_features, out_features=output_features
        )
        layers["softmax"] = nn.Softmax(dim=1)
        return nn.Sequential(layers)

    def forward(self, x):
        """Defines the forward pass of the model."""
        left = self.left(x)
        middle = self.middle(x)
        right = self.right(x)

        concat = torch.cat((left, middle, right), dim=1)
        output = self.fc(concat)

        return output


if __name__ == "__main__":
    # Logging model usage
    logging.info("Classifier model is called".title())

    # Setting up command-line arguments for the model
    parser = argparse.ArgumentParser(description="Classifier model")
    parser.add_argument("--model", help="Model defined")
    args = parser.parse_args()

    # Instantiate and print the model if argument is provided
    if args.model:
        classifier = Classifier()
        print(classifier)

        create_pickle(
            file=classifier,
            filename="/Users/shahmuhammadraditrahman/Desktop/IrisClassifier/models/raw_models.pkl",
        )

    else:
        logging.exception("Model is not defined perfectly".capitalize())
