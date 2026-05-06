"""Tests for model module."""

from src.data_loader import load_data
from src.model import train_model
from src.preprocess import preprocess_data

DATA_PATH = "data/loan_data.csv"


def test_train_model():
    """Test model training."""
    dataframe = load_data(DATA_PATH)

    features, target = preprocess_data(dataframe)

    model, features_test, target_test = train_model(features, target)

    assert model is not None
    assert len(features_test) > 0
    assert len(target_test) > 0
