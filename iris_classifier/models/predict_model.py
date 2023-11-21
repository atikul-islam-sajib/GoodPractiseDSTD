import argparse
import logging
import os
import sys
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

PATH = "/Users/shahmuhammadraditrahman/Desktop/IrisClassifier/iris_classifier"

sys.path.append(PATH)

import config

from utils.utils import load_model, load_pickle, create_pickle

logging.basicConfig(
    level=logging.INFO,
    filemode="w",
    filename="predict.log",
    format="%(levelname)s:%(message)s",
)


class Predict:
    def __init__(self):
        logging.info("Configure the data".title())
        self.model = load_model(
            os.path.join(
                config.MODEL_PATH,
                "model_{}.pth".format(
                    len(os.listdir(os.path.join(config.MODEL_PATH))) - 1
                ),
            )
        )
        self.test_data = load_pickle(
            filename=os.path.join(config.DATA_PATH, "test_loader.pkl")
        )
        self.metrics = {}

    def save_metrics(self, **data):
        logging.info("Save the metrics".title())

        self.metrics["accuracy".title()] = data["accuracy"]
        self.metrics["precision".title()] = data["precision"]
        self.metrics["recall".title()] = data["recall"]
        self.metrics["f1_score".title()] = data["f1_score"]

        try:
            create_pickle(
                file=self.metrics, filename=os.path.join(PATH, "unittest/metrics.pkl")
            )
        except Exception as e:
            logging.exception("Exception in metrics file and caught {}".format(e))

    def predict(self):
        logging.info("Predicting".title())
        actual = []
        pred_labels = []

        for data, label in self.test_data:
            prediction = self.model(data)
            prediction = prediction.argmax(dim=1)
            actual.extend(label)
            pred_labels.extend(prediction)

        logging.info(accuracy_score(actual, pred_labels))
        logging.info(precision_score(actual, pred_labels, average="macro"))
        logging.info(recall_score(actual, pred_labels, average="macro"))
        logging.info(f1_score(actual, pred_labels, average="macro"))

        self.save_metrics(
            accuracy=accuracy_score(actual, pred_labels),
            precision=precision_score(actual, pred_labels, average="macro"),
            recall=recall_score(actual, pred_labels, average="macro"),
            f1_score=f1_score(actual, pred_labels, average="macro"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--predict", action="store_true", help="Model Prediction")

    args = parser.parse_args()

    if args.predict:
        model_predict = Predict()
        model_predict.predict()
    else:
        logging.exception("Exception in the main function of metrics".title())
