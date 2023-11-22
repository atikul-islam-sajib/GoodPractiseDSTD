import argparse
import logging
import sys
import os

PATH = "/Users/shahmuhammadraditrahman/Desktop/IrisClassifier/iris_classifier"
sys.path.append(PATH)

import config
from data.data_loader import DataLoader
from features.build_features import FeatureBuilder
from models.model import Classifier
from models.train_model import Trainer
from models.predict_model import Predict
from utils.utils import load_pickle, create_pickle

logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(config.LOGS_PATH, "command_line.log"),
    filemode="w",
    format="%(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Make --dataset and --split optional and remove the required flag
    parser.add_argument("--dataset", type=str, help="Path of the dataset".title())
    parser.add_argument(
        "--split", type=float, default=0.25, help="Specify the split ratio".title()
    )
    parser.add_argument(
        "--preprocessing",
        action="store_true",
        help="Define the preprocessing of the dataset".title(),
    )
    parser.add_argument(
        "--model", action="store_true", help="Define the Iris model".title()
    )
    parser.add_argument(
        "--epochs", type=int, default=1000, help="Number of epochs".title()
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate".title())
    parser.add_argument("--display", type=bool, default=True, help="Display".title())
    parser.add_argument("--predict", action="store_true", help="Predict".title())

    args = parser.parse_args()

    # Logic for data loading and preprocessing
    if args.preprocessing:
        if args.dataset and args.split:
            logging.info("DataLoader class is called".capitalize())
            try:
                data_loader = DataLoader()
                data_loader.load_data(args.dataset, args.split)
            except ModuleNotFoundError:
                logging.exception("DataLoader class is not found".title())

            logging.info("BuildFeatures class is called".capitalize())
            try:
                feature_builder = FeatureBuilder()
                [
                    feature_builder.build_features(
                        df=os.path.join(config.DATA_PATH, dataset)
                    )
                    for dataset in ["train.csv", "test.csv"]
                ]
                feature_builder.create_data_loader()
            except ModuleNotFoundError:
                logging.exception("BuildFeatures class is not found".title())

            logging.info("DataLoader & FeatureBuilder class is executed".capitalize())

    # Logic for model training
    if args.model:
        if args.epochs and args.lr and args.display:
            logging.info("Model class is called".capitalize())
            try:
                model_trainer = Trainer(args.epochs, args.lr, args.display)
                model_trainer.train()
                create_pickle(
                    file=model_trainer.history,
                    filename=os.path.join(config.MODEL_PATH, "history.pkl"),
                )
                model_trainer.model_performance()
            except ModuleNotFoundError:
                logging.exception("Model class is not found".title())

    if args.predict:
        logging.info("Predict class is called".capitalize())
        try:
            model_predict = Predict()
        except ModuleNotFoundError:
            logging.exception("Predict class is not found".title())
        else:
            logging.info("Predict class is executed".capitalize())

            model_predict.predict()

            metrics = load_pickle(
                filename=os.path.join(config.UNIT_PATH, "metrics.pkl")
            )
            print("Accuracy # {}".upper().format(metrics["Accuracy"]))
            print("Precision # {}".upper().format(metrics["Precision"]))
            print("Recall # {}".upper().format(metrics["Recall"]))
            print("F1_Score # {}".upper().format(metrics["F1_Score"]))

            logging.info("Model performance is printed".capitalize())
