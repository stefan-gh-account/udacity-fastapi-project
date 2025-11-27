
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
import ml.model

# Sample data for testing
X_train = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
y_train = np.array([0, 1, 1, 0])
X_test = np.array([[1, 0], [0, 1]])
y_test = np.array([1, 0])

def test_train_model_returns_randomforest():
    model = ml.model.train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "predict")

def test_inference_returns_correct_shape():
    model = ml.model.train_model(X_train, y_train)
    preds = ml.model.inference(model, X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_test.shape[0]

def test_compute_model_metrics_values():
    # Simulate perfect predictions
    precision, recall, fbeta = ml.model.compute_model_metrics(y_test, y_test)
    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0

def test_compute_model_metrics_with_wrong_preds():
    wrong_preds = np.array([0, 1])  # completely wrong
    precision, recall, fbeta = ml.model.compute_model_metrics(y_test, wrong_preds)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0
