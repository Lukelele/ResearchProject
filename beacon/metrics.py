import torch


###########################   Accuracy Metrics   ###########################

def categorical_accuracy(y_pred_logit, y_true):
    """
    Calculates the categorical accuracy of the predicted logits.

    Args:
    - y_pred_logit: predicted logits
    - y_true: true labels

    Returns:
    - accuracy: categorical accuracy of the predicted logits
    """
    y_prob = torch.softmax(y_pred_logit, dim=1)
    y_pred = torch.argmax(y_prob, dim=1)
    correct = (y_pred == y_true).float()
    accuracy = correct.sum() / len(correct)

    return accuracy


def binary_accuracy(y_pred_logit, y_true):
    """
    Calculates specifically the binary accuracy of the predicted logits.

    Args:
    - y_pred_logit: predicted logits
    - y_true: true labels

    Returns:
    - accuracy: binary accuracy of the predicted logits
    """
    y_prob = torch.sigmoid(y_pred_logit)
    y_pred = torch.argmax(y_prob)
    correct = (y_pred == y_true).float()
    accuracy = correct.sum() / len(correct)

    return accuracy


def precision(y_pred_logit, y_true):
    """
    Calculates the precision of the predicted logits.

    Args:
    - y_pred_logit: predicted logits
    - y_true: true labels

    Returns:
    - precision: precision of the predicted logits
    """
    y_prob = torch.sigmoid(y_pred_logit)
    y_pred = torch.round(y_prob)

    true_positive = torch.sum(y_pred * y_true)
    false_positive = torch.sum(y_pred * (1 - y_true))

    precision = true_positive / (true_positive + false_positive)

    return precision


def recall(y_pred_logit, y_true):
    """
    Calculates the recall of the predicted logits.

    Args:
    - y_pred_logit: predicted logits
    - y_true: true labels

    Returns:
    - recall: recall of the predicted logits
    """
    y_prob = torch.sigmoid(y_pred_logit)
    y_pred = torch.round(y_prob)

    true_positive = torch.sum(y_pred * y_true)
    false_negative = torch.sum((1 - y_pred) * y_true)

    recall = true_positive / (true_positive + false_negative)

    return recall


def F1(y_pred_logit, y_true):
    """
    Calculates the F1 score of the predicted logits.

    Args:
    - y_pred_logit: predicted logits
    - y_true: true labels

    Returns:
    - F1: F1 score of the predicted logits
    """
    precision_score = precision(y_pred_logit, y_true)
    recall_score = recall(y_pred_logit, y_true)

    F1 = 2 * (precision_score * recall_score) / (precision_score + recall_score)

    return F1


###########################   Regression Metrics   ###########################





##################################   Misc   ##################################

# Define a lookup table in case the user wants to use a string to specify the metric
lookup_table = {
    "categorical_accuracy": categorical_accuracy,
    "binary_accuracy": binary_accuracy,
    "precision": precision,
    "recall": recall,
    "F1": F1
}
