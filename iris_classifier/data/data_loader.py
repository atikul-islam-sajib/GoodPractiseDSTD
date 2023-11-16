from utils.utils import create_pickle
import argparse
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

import sys

sys.path.append("/Users/shahmuhammadraditrahman/Desktop/IrisClassifier/iris_classifier")


logging.basicConfig(
    level=logging.INFO,
    filename="dataset.log",
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
        logging.info("Loading data")
        self.dataset = dataset
        self.split_ratio = split

        # Split the dataset
        self.split_dataset(dataset=self.dataset, split_ratio=self.split_ratio)

    def split_dataset(self, **dataset):
        """
        Split the loaded dataset into training and testing sets, and save them as CSV files.

        Raises:
            ValueError: If the dataset is not loaded.

        Returns:
            None
        """
        data_frame = pd.read_csv(dataset["dataset"])
        split_ratio = dataset["split_ratio"]

        logging.info("Splitting dataset")
        train, test = train_test_split(
            data_frame, test_size=split_ratio, random_state=42
        )
        try:
            train.to_csv(
                "/Users/shahmuhammadraditrahman/Desktop/IrisClassifier/data/processed/train.csv",
                index=False,
            )
            test.to_csv(
                "/Users/shahmuhammadraditrahman/Desktop/IrisClassifier/data/processed/test.csv",
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
        "--split", type=float, default=0.20, help="Split ratio of the dataset"
    )

    args = parser.parse_args()

    if args.dataset:
        data_loader = DataLoader()
        data_loader.load_data(args.dataset, args.split)

        logging.info("Completed the data splitting and loader".capitalize())
    else:
        logging.exception("Dataset path is not found".title())
