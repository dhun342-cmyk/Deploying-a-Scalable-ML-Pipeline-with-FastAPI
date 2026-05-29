# TODO: add necessary import

# import pytest
import numpy as np
from fastapi.testclient import TestClient
from main import app
from ml.model import compute_model_metrics
from sklearn.ensemble import RandomForestClassifier
client = TestClient(app)


# TODO: implement the first test. Change the function name and input as needed


def test_post_inference():
    """
    # test the POST endpoint for model inference
    """
    sample = {
        "age": 37,
        "workclass": "Private",
        "fnlgt": 178356,
        "education": "HS-grad",
        "education-num": 10,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }

    response = client.post("/data/", json=sample)
    assert response.status_code == 200
    body = response.json()
    assert "result" in body
    assert body["result"] in [">50K", "<=50K"]


# TODO: implement the second test. Change the function name and input as needed


def data_type_validation(X_train, y_train, X_test, y_test):
    """
    # verify that the training and test datasets
    # have the expected size and data types
    """
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)


# TODO: implement the third test. Change the function name and input as neededS


def pipeline_validation(X_train, y_train, X_test, y_test):
    """
    # validate the end-to-end machine learning pipeline
    # by training a model and evaluating its performance
    # on the test dataset
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert precision >= 0.5
    assert recall >= 0.5
    assert fbeta >= 0.5
