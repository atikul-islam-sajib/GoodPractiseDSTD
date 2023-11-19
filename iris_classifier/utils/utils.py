import joblib
import torch.nn as nn
import torch.optim as optim


def create_pickle(file, filename):
    joblib.dump(value=file, filename=filename)


def load_pickle(filename):
    return joblib.load(filename=filename)


def define_optimizer(model, lr=0.001):
    return optim.AdamW(params=model.parameters(), lr=lr)


def define_loss_function():
    return nn.CrossEntropyLoss()



