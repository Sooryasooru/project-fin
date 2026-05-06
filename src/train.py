"""Training pipeline module."""

import joblib

from src.data_loader import load_data
from src.evaluate import evaluate_model
from src.model import train_model
from src.preprocess import preprocess_data

DATA_PATH = "data/loan_data.csv"
MODEL_PATH = "models/model.pkl"


def main():
    """Run complete loan repayment prediction pipeline."""
    dataframe = load_data(DATA_PATH)

    features, target = preprocess_data(dataframe)

    model, features_test, target_test = train_model(features, target)

    accuracy, report, matrix = evaluate_model(
        model,
        features_test,
        target_test,
    )

    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:\n")
    print(report)
    print("\nConfusion Matrix:\n")
    print(matrix)

    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
