"""Tests for evaluation module."""

from src.data_loader import load_data
from src.evaluate import evaluate_model
from src.model import train_model
from src.preprocess import preprocess_data

DATA_PATH = "data/loan_data.csv"


def test_evaluate_model():
    """Test model evaluation."""
    dataframe = load_data(DATA_PATH)

    features, target = preprocess_data(dataframe)

    model, features_test, target_test = train_model(features, target)

    accuracy, report, matrix = evaluate_model(
        model,
        features_test,
        target_test,
    )

    assert accuracy >= 0
    assert report is not None
    assert matrix is not None
