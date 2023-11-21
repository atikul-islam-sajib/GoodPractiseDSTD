import logging
import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    filename="features.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

import sys

sys.path.append("/Users/shahmuhammadraditrahman/Desktop/IrisClassifier/iris_classifier")

from utils.utils import create_pickle, load_pickle

PATH = "/Users/shahmuhammadraditrahman/Desktop/IrisClassifier/data/processed/"


class FeatureBuilder:
    """
    A class for building features from a given dataset.

    Attributes:
        None

    Methods:
        __init__: Initializes the FeatureBuilder instance.
        build_features: Builds features from a given dataset.
        _label_encoding: Performs label encoding on the 'species' column.
        _normalization: Performs standard scaling for normalization.

    Usage:
        feature_builder = FeatureBuilder()
        feature_builder.build_features(df)
    """

    def __init__(self):
        logger.info("Initializing FeatureBuilder")

    def build_features(self, df):
        """
        Builds features from a given dataset.

        Args:
            df (str): The path to the input CSV file.

        Returns:
            None
        """
        logging.info("Building Features starts")
        data_frame = pd.read_csv(df)

        logging.info("Creating the pickle file")
        create_pickle(
            file=data_frame,
            filename="/Users/shahmuhammadraditrahman/Desktop/IrisClassifier/data/processed/{}.pkl".format(
                df.split("/")[-1].split(".")[0]
            ),
        )

    def create_data_loader(self):
        """
        Loads training and validation datasets from pickled files and converts them into tensors.

        This method performs the following steps:
        1. Loads the training dataset from 'train.pkl' and the validation dataset from 'test.pkl'.
        2. These datasets are expected to be in a predefined directory specified by the PATH variable.
        3. Each dataset is then converted into a tensor format suitable for model training and evaluation.
        4. The tensors are stored as 'train_loader' and 'test_loader' for training and validation datasets, respectively.

        The method does not return any values but updates the instance attributes related to data loaders.
        """
        train_dataset = load_pickle(filename=os.path.join(PATH, "train.pkl"))
        val_dataset = load_pickle(filename=os.path.join(PATH, "test.pkl"))
        [
            self._convert_into_tensor(
                data=dataset, name="train_loader" if index == 0 else "test_loader"
            )
            for index, dataset in enumerate([train_dataset, val_dataset])
        ]

    def _convert_into_tensor(self, **dataset):
        """
        Converts a given dataset into a tensor and saves it as a pickle file.

        Parameters:
        - dataset (dict): A dictionary containing the dataset and its name.
                        The dataset is expected to have 'data' and 'name' keys.
                        'data' should be a pandas DataFrame with the last column as the target variable.
                        'name' is a string used for naming the output file.

        Steps:
        1. Splits the dataset into features (X) and the target variable (y).
        2. Converts the features (X) into a tensor of type torch.float32.
        3. Calls the '_tensor_to_dataloader' method to convert the tensor into a data loader format.
        4. Saves the data loader as a pickle file in the specified directory. The file name is derived from the dataset name.

        The method does not return any values but performs data processing and saving operations.
        """
        X, y = dataset["data"].iloc[:, :-1].values, dataset["data"].iloc[:, -1].values

        X = torch.tensor(data=X, dtype=torch.float32)

        loader = self._tensor_to_dataloader(X=X, y=y)

        create_pickle(
            file=loader, filename=os.path.join(PATH, "{}.pkl".format(dataset["name"]))
        )

    def _tensor_to_dataloader(self, **dataset):
        """
        Converts tensors of features and labels into a DataLoader object.

        This method is used to create a DataLoader from the given feature and label tensors.
        The DataLoader is a PyTorch utility that allows for efficient batching, shuffling,
        and loading of data during model training and evaluation.

        Parameters:
        - dataset (dict): A dictionary containing the features and labels.
                        It must have keys 'X' and 'y', where 'X' is a tensor of features
                        and 'y' is a tensor of labels.

        Returns:
        - data_loader (DataLoader): A DataLoader object created from the given tensors.
                                    It batches the data with a specified batch size (default is 64).

        The method takes the features and labels, zips them into a single dataset,
        and then creates a DataLoader object for efficient data handling.
        """
        data_loader = DataLoader(
            dataset=list(zip(dataset["X"], dataset["y"])), batch_size=16
        )
        return data_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--preprocessing", type=str, help="Feature Engineering", default=True
    )

    args = parser.parse_args()

    feature_builder = FeatureBuilder()
    if args.preprocessing:
        data_path = (
            "/Users/shahmuhammadraditrahman/Desktop/IrisClassifier/data/processed/"
        )
        [
            feature_builder.build_features(df=os.path.join(data_path, dataset))
            for dataset in ["train.csv", "test.csv"]
        ]
        feature_builder.create_data_loader()
    else:
        logging.exception("File not found".title())
