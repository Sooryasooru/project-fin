"""Data preprocessing module."""


def preprocess_data(dataframe):
    """Prepare features and target variables."""
    features = dataframe.drop(columns=["loan_status", "customer_id"])
    target = dataframe["loan_status"]

    return features, target
