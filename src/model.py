"""Model training module."""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def train_model(features, target):
    """Train logistic regression model."""
    features_train, features_test, target_train, target_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(features_train, target_train)

    return model, features_test, target_test
