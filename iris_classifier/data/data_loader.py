import argparse
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
import os

sys.path.append("./iris_classifier")

from utils.utils import create_pickle
import config

logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(config.LOGS_PATH, "dataset.log"),
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class DataLoader:
    def __init__(self):
        pass

    def load_data(self, dataset="IrisClassifier/data/raw/IRIS.csv", split=0.20):
        """
        Load data from a CSV file and store it in the DataLoader instance.

        Parameters:
            dataset_path (str): The file path of the dataset to load.

        Returns:
            None
        """
        logging.info("Loading data & Scaling the dataset".title())
        self.dataset = pd.read_csv(dataset)
        self.split_ratio = split

        self.dataset.loc[:, "species"] = self.dataset.loc[:, "species"].map(
            self._label_encoding(self.dataset)
        )

        dataset = self._normalization(dataset=self.dataset)
        self.split_dataset(
            dataset=self.dataset, split_ratio=self.split_ratio, random_state=42
        )

    def _label_encoding(self, data_frame):
        """
        Performs label encoding on the 'species' column.

        Args:
            target (pd.Series): The target column to encode.

        Returns:
            dict: A dictionary mapping unique values to indices.
        """
        return {
            value: index
            for index, value in enumerate(data_frame.iloc[:, -1].value_counts().index)
        }

    def _normalization(self, dataset):
        """
        Performs standard scaling for normalization.

        Args:
            dataset (pd.DataFrame): The input dataset.

        Returns:
            pd.DataFrame: The normalized dataset.
        """
        logging.info("Standard scaling is used for normalization technique".title())

        scaler = StandardScaler()
        independent_features = scaler.fit_transform(
            dataset.loc[:, dataset.columns != "species"]
        )

        dependent_features = dataset.loc[:, "species"]
        independent_features = pd.DataFrame(independent_features)
        dependent_features = pd.DataFrame(dependent_features)
        return pd.concat([independent_features, dependent_features], axis=1)

    def split_dataset(self, **dataset):
        """
        Split the loaded dataset into training and testing sets, and save them as CSV files.

        Raises:
            ValueError: If the dataset is not loaded.

        Returns:
            None
        """
        data_frame = dataset["dataset"]
        split_ratio = dataset["split_ratio"]

        logging.info("Splitting dataset")
        train, test = train_test_split(data_frame, test_size=split_ratio)
        try:
            train.to_csv(
                os.path.join(config.DATA_PATH, "train.csv"),
                index=False,
            )
            test.to_csv(
                os.path.join(config.DATA_PATH, "test.csv"),
                index=False,
            )
        except ValueError as e:
            logging.exception("File {} path is not found".title().format(e))


if __name__ == "__main__":
    """
    Main function to parse command-line arguments, load and split the dataset using DataLoader.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="IrisClassifier/data/raw/IRIS.csv",
        help="Dataset path",
    )
    parser.add_argument(
        "--split", type=float, default=0.25, help="Split ratio of the dataset"
    )

    args = parser.parse_args()

    if args.dataset:
        data_loader = DataLoader()
        data_loader.load_data(args.dataset, args.split)

        logging.info("Completed the data splitting and loader".capitalize())
    else:
        logging.exception("Dataset path is not found".title())
