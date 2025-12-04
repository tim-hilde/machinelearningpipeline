import os
import pickle
from urllib.parse import urlparse

import mlflow
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score

os.environ["MLFLOW_TRACKING_URI"] = (
    "https://dagshub.com/tim-hilde/machinelearningpipeline.mlflow"
)
os.environ["MLFLOW_TRACKING_USERNAME"] = "tim-hilde"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "8b8b78a74a69c016722c94f713e25b27bfa7fc29"


# Load parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))["train"]


def evaluate(data_path, model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri(
        "https://dagshub.com/tim-hilde/machinelearningpipeline.mlflow"
    )

    ## load the model from the disk
    model = pickle.load(open(model_path, "rb"))

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    ## log metrics to MLFLOW

    mlflow.log_metric("accuracy", accuracy)
    print("Model accuracy:{accuracy}")


if __name__ == "__main__":
    evaluate(params["data"], params["model"])
