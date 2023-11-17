import logging
import argparse
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    filename="train_model.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
"""
trainer = Trainer(epochs = 100, lr = 0.001)
"""


class Trainer:
    def __init__(self, epochs=100, lr=0.001):
        self.epochs = epochs
        self.learning_rate = lr

    def something(self):
        pass
