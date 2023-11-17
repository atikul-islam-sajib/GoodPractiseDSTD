import logging
import argparse
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure basic logging settings
logging.basicConfig(
    level=logging.INFO,
    filename="chars.log",
    filemode="w",
    format="%(levelname)s %(asctime)s %(message)s",
)

# Path where the charts will be saved
PATH = "/Users/shahmuhammadraditrahman/Desktop/IrisClassifier/iris_classifier/visualization/charts"


class Visualizer:
    """A class for visualizing various aspects of a dataset."""

    def __init__(self):
        """Initialize the Visualizer class."""
        logging.info("Initializing Visualizer class")

    def get_all_charts(self):
        """Read dataset and generate all charts."""
        # Load the dataset
        data_frame = pd.read_csv(
            "/Users/shahmuhammadraditrahman/Desktop/IrisClassifier/data/raw/IRIS.csv"
        )

        # Generate various charts
        self._find_correlation_between_features(dataset=data_frame)
        self._show_distribution_target_class(dataset=data_frame)
        self._check_outliers(dataset=data_frame)
        self._show_distribution(dataset=data_frame)

    def _find_correlation_between_features(self, dataset):
        """Generate and save a heatmap showing correlations between features."""

        logging.info("Correlation between dataset is generating".title)

        plt.title("Correlation between features")
        sns.heatmap(
            dataset.loc[:, dataset.columns[:-1]].corr(), annot=True, cmap="YlGnBu"
        )
        plt.savefig("{}correlation.png".format(PATH), format="png")
        plt.clf()

    def _show_distribution_target_class(self, dataset):
        """Generate and save a bar chart showing the distribution of the target class."""

        logging.info("Distribution of target class is generating".title)
        dataset.loc[:, dataset.columns[-1]].value_counts().plot(kind="bar")

        plt.xlabel("Target Class")
        plt.ylabel("Count")
        plt.title("Distribution of Target Class")

        plt.savefig("{}target_distribution.png".format(PATH))
        plt.clf()

    def _check_outliers(self, dataset):
        """Generate and save boxplots to visualize outliers in each feature."""
        logging.info("Outliers is generating".title)

        plt.title("Boxplots of Features")
        for index, feature in enumerate(dataset.loc[:, dataset.columns[:-1]]):
            plt.subplot(1, 4, index + 1)
            sns.boxplot(y=feature, data=dataset)

        plt.savefig("{}outliers.png".format(PATH))
        plt.clf()

    def _show_distribution(self, dataset):
        """Generate and save histograms showing the distribution of each feature."""
        logging.info("Distribution between features is generating".title)

        for index, feature in enumerate(dataset.loc[:, dataset.columns[:-1]]):
            plt.subplot(1, 4, index + 1)
            sns.histplot(data=dataset, x=feature)

        plt.savefig("{}distribution.png".format(PATH))
        plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Visualization")

    parser.add_argument("--charts", help="Show the plot", default=True)

    args = parser.parse_args()

    if args.charts:
        charts = Visualizer()
        charts.get_all_charts()
    else:
        logging.exception("Charts cannot be displayed.".title())
