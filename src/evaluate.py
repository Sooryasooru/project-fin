"""Model evaluation module."""

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def evaluate_model(model, features_test, target_test):
    """Evaluate model performance."""
    predictions = model.predict(features_test)

    accuracy = accuracy_score(target_test, predictions)
    report = classification_report(target_test, predictions)
    matrix = confusion_matrix(target_test, predictions)

    return accuracy, report, matrix
