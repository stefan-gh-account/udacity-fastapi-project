"""Script to train machine learning model."""

import sys
import os
import numpy
import pandas
import sklearn.model_selection
import ml.data
import ml.model
import joblib


data_path = "data/cleaned_census.csv"
model_path = "model/model.pkl"


def clean_csv(original_file_path: str, cleaned_file_path: str):
    """Remove spaces from a CSV file."""
    with open(original_file_path, "r") as f:
        text = f.read()
    text = text.replace(" ", "")
    with open(cleaned_file_path, "w") as f:
        f.write(text)


if __name__ == "__main__":
    # remove spaces from csv file
    clean_csv("data/census.csv", data_path)
    data = pandas.read_csv(data_path)
    data = data.drop("education-num", axis=1)  # redundant with education

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = sklearn.model_selection.train_test_split(data, test_size=0.20)

    X_train, y_train, encoder, lb = ml.data.process_data(
        train, categorical_features=ml.data.categorical_features, label="salary", training=True)
    X_test, y_test, _, _ = ml.data.process_data(
        test, categorical_features=ml.data.categorical_features, label="salary", training=False, 
        encoder=encoder, lb=lb)

    model = ml.model.train_model(X_train, y_train)
    joblib.dump({"model": model,
                 "encoder": encoder,
                 "label_binarizer": lb},
                "model/model_dict.pkl")

    # Computes model performance metrics for different slices of a feature.
    for feature in ml.data.categorical_features:
        unique_values = test[feature].unique()
        for value in unique_values:
            idx = test[feature] == value
            print(f"Feature: {feature}={value} ({100 * sum(idx) / len(idx):.1f}%, "
                  f"{sum(idx)} persons) | ", end="")
            if not any(idx):
                print("No samples found.")
            else:
                # print(X_train.shape, X_test.shape, X_test[idx].shape)
                predictions = model.predict(X_test[idx])
                precision, recall, fbeta = ml.model.compute_model_metrics(y_test[idx], predictions)
                print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {fbeta:.4f}")

    predictions = model.predict(X_test)
    precision, recall, fbeta = ml.model.compute_model_metrics(y_test, predictions)
    print(f"\nOverall performance: Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {fbeta:.4f}")
