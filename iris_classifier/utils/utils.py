import joblib


def create_pickle(file, filename):
    joblib.dump(value=file, filename=filename)


def load_pickle(filename):
    return joblib.load(filename=filename)
