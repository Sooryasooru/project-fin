"""Tests for preprocess module."""

from src.data_loader import load_data
from src.preprocess import preprocess_data

DATA_PATH = "data/loan_data.csv"


def test_preprocess_data():
    """Test preprocessing functionality."""
    dataframe = load_data(DATA_PATH)

    features, target = preprocess_data(dataframe)

    assert features is not None
    assert target is not None
    assert "loan_status" not in features.columns
