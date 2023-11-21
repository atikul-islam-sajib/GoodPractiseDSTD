import unittest
import sys
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    filename="unit_test.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

sys.path.append("/Users/shahmuhammadraditrahman/Desktop/IrisClassifier/iris_classifier")

from utils.utils import load_pickle

PATH = "/Users/shahmuhammadraditrahman/Desktop/IrisClassifier/data/processed/"
METRICS_PATH = (
    "/Users/shahmuhammadraditrahman/Desktop/IrisClassifier/iris_classifier/unittest"
)


class FeatureTest(unittest.TestCase):
    """
    A test case for validating the feature data loaders for the Iris Classifier project.

    This class contains tests to ensure that the training and validation datasets loaded from
    pickle files have the correct number of records as expected.
    """

    def setUp(self):
        """
        Set up test environment.

        Loads the training and validation datasets from pickle files before each test.
        Initializes the total_records counter to 0.
        """

        logging.info("Running FeatureTest setUp")

        self.train_loader = load_pickle(
            filename=os.path.join(PATH, "train_loader.pkl"),
        )
        self.test_loader = load_pickle(
            filename=os.path.join(PATH, "test_loader.pkl"),
        )
        self.total_records = 0

    def tearDown(self):
        """
        Tear down the test environment.

        Resets the total_records counter to 0 after each test.
        """
        logging.info("Running FeatureTest tearDown")

        self.total_records = 0

    def test_train_features_pickle(self):
        """
        Test to ensure the training dataset has the correct number of records.

        Iterates through the training data loader and sums up the number of records.
        Asserts that this sum is approximately equal to the expected number (120).
        """

        logging.info("Running FeatureTest test_train_features_pickle")

        for data, _ in self.train_loader:
            self.total_records += data.shape[0]

        self.assertAlmostEqual(self.total_records, 120)

    def test_val_features_pickle(self):
        """
        Test to ensure the validation dataset has the correct number of records.

        Iterates through the validation data loader and sums up the number of records.
        Asserts that this sum is exactly equal to the expected number (30).
        """

        logging.info("Running FeatureTest test_val_features_pickle")

        for data, _ in self.test_loader:
            self.total_records += data.shape[0]

        self.assertEqual(self.total_records, 30)


class ModelPrediction(unittest.TestCase):
    def setUp(self):
        self.metrics = load_pickle(filename=os.path.join(METRICS_PATH, "metrics.pkl"))
        print(self.metrics)

    def test_predict_model(self):
        self.assertGreater(self.metrics["Accuracy"], 0.80)
        self.assertGreater(self.metrics["Precision"], 0.80)
        self.assertGreater(self.metrics["Recall"], 0.80)
        self.assertGreater(self.metrics["F1_Score"], 0.80)


if __name__ == "__main__":
    unittest.main()
