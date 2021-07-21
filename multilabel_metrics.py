import numpy as np


def cal_multilabel_accuracy(targets, outputs, average="samples", **kwargs):
    assert average in ["samples", "macro", "micro"], "average much be either 'samples', 'macro', 'micro'"

    if average == "samples":
        numerator = np.sum(np.logical_and(targets, outputs), axis=1)
        denominator = np.sum(np.logical_or(targets, outputs), axis=1)
        avg_accuracy = np.mean(numerator / (denominator + 1e-7))
    elif average == "macro":
        numerator = np.sum(np.logical_and(targets, outputs), axis=0)
        denominator = np.sum(np.logical_or(targets, outputs), axis=0)
        avg_accuracy = np.mean(numerator / (denominator + 1e-7))
    else:
        numerator = np.sum(np.logical_and(targets, outputs))
        denominator = np.sum(np.logical_or(targets, outputs))
        avg_accuracy = numerator / (denominator + 1e-7)
    return avg_accuracy


def cal_multilabel_precision(targets, outputs, average="samples", **kwargs):
    assert average in ["samples", "macro", "micro"], "average much be either 'samples', 'macro', 'micro'"

    if average == "samples":
        numerator = np.sum(np.logical_and(targets, outputs), axis=1)
        denominator = np.sum(outputs, axis=1)
        avg_precision = np.mean(numerator / (denominator + 1e-7))
    elif average == "macro":
        numerator = np.sum(np.logical_and(targets, outputs), axis=0)
        denominator = np.sum(outputs, axis=0)
        avg_precision = np.mean(numerator / (denominator + 1e-7))
    else:
        numerator = np.sum(np.logical_and(targets, outputs))
        denominator = np.sum(outputs)
        avg_precision = numerator / (denominator + 1e-7)
    return avg_precision


def cal_multilabel_recall(targets, outputs, average="samples", **kwargs):
    assert average in ["samples", "macro", "micro"], "average much be either 'samples', 'macro', 'micro'"

    if average == "samples":
        numerator = np.sum(np.logical_and(targets, outputs), axis=1)
        denominator = np.sum(targets, axis=1)
        avg_recall = np.mean(numerator / (denominator + 1e-7))
    elif average == "macro":
        numerator = np.sum(np.logical_and(targets, outputs), axis=0)
        denominator = np.sum(targets, axis=0)
        avg_recall = np.mean(numerator / (denominator + 1e-7))
    else:
        numerator = np.sum(np.logical_and(targets, outputs))
        denominator = np.sum(targets)
        avg_recall = numerator / (denominator + 1e-7)
    return avg_recall


def cal_multilabel_f1(targets, outputs, average="samples", **kwargs):
    precision = cal_multilabel_precision(targets, outputs, average=average)
    recall = cal_multilabel_recall(targets, outputs, average=average)
    return 2 * ((precision * recall) / (precision + recall + 1e-7))
