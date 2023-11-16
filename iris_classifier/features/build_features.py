import logging
import argparse
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    filename="features.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

import sys

sys.path.append("/Users/shahmuhammadraditrahman/Desktop/IrisClassifier/iris_classifier")

from utils.utils import create_pickle


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
        data_frame.loc[:, "species"] = data_frame.loc[:, "species"].map(
            self._label_encoding(data_frame.loc[:, "species"])
        )
        dataset = self._normalization(dataset=data_frame)

        logging.info("Creating the pickle file")
        create_pickle(
            file=dataset,
            filename="/Users/shahmuhammadraditrahman/Desktop/IrisClassifier/data/processed/{}.pkl".format(
                df.split("/")[-1].split(".")[0]
            ),
        )

    def _label_encoding(self, target):
        """
        Performs label encoding on the 'species' column.

        Args:
            target (pd.Series): The target column to encode.

        Returns:
            dict: A dictionary mapping unique values to indices.
        """
        return {value: index for index, value in enumerate(target.unique())}

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
    else:
        logging.exception("File not found".title())
