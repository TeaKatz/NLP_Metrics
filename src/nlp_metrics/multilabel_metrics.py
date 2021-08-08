import numpy as np


def cal_multilabel_accuracy(inputs, targets, average="samples", **kwargs):
    """
    targets: (batch_size, vector_size)
    inputs: (batch_size, vector_size)
    """
    assert average in ["samples", "macro", "micro"], "average much be either 'samples', 'macro', 'micro'"

    if average == "samples":
        numerator = np.sum(np.logical_and(inputs, targets), axis=1)
        denominator = np.sum(np.logical_or(inputs, targets), axis=1)
        avg_accuracy = np.mean(numerator / (denominator + 1e-7))
    elif average == "macro":
        numerator = np.sum(np.logical_and(inputs, targets), axis=0)
        denominator = np.sum(np.logical_or(inputs, targets), axis=0)
        avg_accuracy = np.mean(numerator / (denominator + 1e-7))
    else:
        numerator = np.sum(np.logical_and(inputs, targets))
        denominator = np.sum(np.logical_or(inputs, targets))
        avg_accuracy = numerator / (denominator + 1e-7)
    return avg_accuracy


def cal_multilabel_precision(inputs, targets, average="samples", **kwargs):
    """
    targets: (batch_size, vector_size)
    inputs: (batch_size, vector_size)
    """
    assert average in ["samples", "macro", "micro"], "average much be either 'samples', 'macro', 'micro'"

    if average == "samples":
        numerator = np.sum(np.logical_and(inputs, targets), axis=1)
        denominator = np.sum(inputs, axis=1)
        avg_precision = np.mean(numerator / (denominator + 1e-7))
    elif average == "macro":
        numerator = np.sum(np.logical_and(inputs, targets), axis=0)
        denominator = np.sum(inputs, axis=0)
        avg_precision = np.mean(numerator / (denominator + 1e-7))
    else:
        numerator = np.sum(np.logical_and(inputs, targets))
        denominator = np.sum(inputs)
        avg_precision = numerator / (denominator + 1e-7)
    return avg_precision


def cal_multilabel_recall(inputs, targets, average="samples", **kwargs):
    """
    targets: (batch_size, vector_size)
    inputs: (batch_size, vector_size)
    """
    assert average in ["samples", "macro", "micro"], "average much be either 'samples', 'macro', 'micro'"

    if average == "samples":
        numerator = np.sum(np.logical_and(inputs, targets), axis=1)
        denominator = np.sum(targets, axis=1)
        avg_recall = np.mean(numerator / (denominator + 1e-7))
    elif average == "macro":
        numerator = np.sum(np.logical_and(inputs, targets), axis=0)
        denominator = np.sum(targets, axis=0)
        avg_recall = np.mean(numerator / (denominator + 1e-7))
    else:
        numerator = np.sum(np.logical_and(inputs, targets))
        denominator = np.sum(targets)
        avg_recall = numerator / (denominator + 1e-7)
    return avg_recall


def cal_multilabel_f1(inputs, targets, average="samples", **kwargs):
    """
    targets: (batch_size, vector_size)
    inputs: (batch_size, vector_size)
    """
    precision = cal_multilabel_precision(inputs, targets, average=average)
    recall = cal_multilabel_recall(inputs, targets, average=average)
    return 2 * ((precision * recall) / (precision + recall + 1e-7))
