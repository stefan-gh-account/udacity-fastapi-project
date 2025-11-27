"""Script to train machine learning model."""

import sys
import os
import pandas
import sklearn.model_selection
from ml.data import process_data
from ml.model import train_model
import joblib


data_path = "data/cleaned_census.csv"


if __name__ == "__main__":
    # remove spaces from csv file
    if not os.path.isfile(data_path):
        with open("data/census.csv", "r") as f:
            text = f.read()
        text = text.replace(" ", "")
        with open(data_path, "w") as f:
            f.write(text)

    data = pandas.read_csv(data_path)

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = sklearn.model_selection.train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    model = train_model(X_train, y_train)
    joblib.dump(model, "model/model.pkl")
    