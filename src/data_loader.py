"""Data loading module."""

import pandas as pd

REQUIRED_COLUMNS = [
    "customer_id",
    "age",
    "annual_income",
    "loan_amount",
    "credit_score",
    "employment_years",
    "existing_debt",
    "loan_term_months",
    "missed_payments",
    "last_transaction_balance",
    "loan_status",
]


def load_data(path):
    """Load loan repayment dataset from CSV file."""
    dataframe = pd.read_csv(path)

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    return dataframe
