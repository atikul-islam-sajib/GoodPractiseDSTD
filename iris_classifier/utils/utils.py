import joblib
import pandas as pd
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def create_pickle(file, filename):
    joblib.dump(value=file, filename=filename)


def load_pickle(filename):
    return joblib.load(filename=filename)


def define_optimizer(model, lr=0.001):
    return optim.AdamW(params=model.parameters(), lr=lr)


def define_loss_function():
    return nn.CrossEntropyLoss()


def load_model(model_path):
    return torch.load(model_path)


def get_data(data_path):
    data = pd.read_csv(data_path)
    data.iloc[:, -1] = data.iloc[:, -1].map(
        {
            value: index
            for index, value in enumerate(data.iloc[:, -1].value_counts().index)
        }
    )
    data, label = data.iloc[:, :-1].values, data.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(
        data, label, test_size=0.2, random_state=42
    )
    X_train = torch.tensor(data=X_train, dtype=torch.float32)
    X_test = torch.tensor(data=X_test, dtype=torch.float32)

    train_loader = DataLoader(dataset=list(zip(X_train, y_train)), batch_size=16)
    test_loader = DataLoader(dataset=list(zip(X_test, y_test)), batch_size=16)

    return train_loader, test_loader
