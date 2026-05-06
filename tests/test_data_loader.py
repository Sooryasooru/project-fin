"""Tests for data loader module."""

from src.data_loader import load_data

DATA_PATH = "data/loan_data.csv"


def test_load_data():
    """Test dataset loading."""
    dataframe = load_data(DATA_PATH)

    assert dataframe is not None
    assert dataframe.empty is False
