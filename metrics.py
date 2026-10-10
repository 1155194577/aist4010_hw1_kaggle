from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def evaluate_model(y_true, y_pred):
    """
    Evaluate the model's performance using various metrics.

    Args:
        y_true (list or array): True labels.
        y_pred (list or array): Predicted labels.

    Returns:
        dict: A dictionary containing accuracy, F1 score, confusion matrix, and classification report.
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, zero_division=0)

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "confusion_matrix": conf_matrix,
        "classification_report": class_report
    }