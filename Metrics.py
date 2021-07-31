import fastwer
import numpy as np

from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from .multilabel_metrics import cal_multilabel_accuracy, cal_multilabel_precision, cal_multilabel_recall, cal_multilabel_f1


def cal_cer(outputs, targets, **kwargs):
    """
    targets: List(String)
    outputs: List(String)
    """
    return fastwer.score(targets, outputs, char_level=True)


def cal_wer(outputs, targets, **kwargs):
    """
    targets: List(String)
    outputs: List(String)
    """
    return fastwer.score(targets, outputs)


def cal_bleu(outputs, targets, **lwargs):
    """
    targets: List(List(String))
    outputs: List(List(String))
    """
    bleu_score = np.mean([sentence_bleu([tar_tokens], out_tokens) for tar_tokens, out_tokens in zip(targets, outputs)])
    return bleu_score


def cal_cosine_similarity(outputs, targets, **kwargs):
    """
    target: (batch_size, vector_size)
    outputs: (batch_size, vector_size)
    """
    return cosine_similarity(targets, outputs).diagonal().mean()


def cal_accuracy(outputs, targets, **kwargs):
    """
    targets: (batch_size, vector_size)
    outputs: (batch_size, vector_size)
    """
    return accuracy_score(targets, outputs)


def cal_precision(outputs, targets, labels=None, pos_label=1, average="binary", sample_weight=None, zero_division="warn", **kwargs):
    """
    targets: (batch_size, vector_size)
    outputs: (batch_size, vector_size)
    """
    return precision_score(targets, outputs, labels=labels, average=average, zero_division=zero_division)


def cal_recall(outputs, targets, labels=None, pos_label=1, average="binary", sample_weight=None, zero_division="warn", **kwargs):
    """
    targets: (batch_size, vector_size)
    outputs: (batch_size, vector_size)
    """
    return recall_score(targets, outputs, labels=labels, average=average)


def cal_f1(outputs, targets, labels=None, pos_label=1, average="binary", sample_weight=None, zero_division="warn", **kwargs):
    """
    targets: (batch_size, vector_size)
    outputs: (batch_size, vector_size)
    """
    return f1_score(targets, outputs, labels=labels, average=average)


def cal_mae(outputs, targets):
    """
    targets: (batch_size, vector_size)
    outputs: (batch_size, vector_size)
    """
    return np.mean(np.abs(targets - outputs))


def cal_mse(outputs, targets):
    """
    targets: (batch_size, vector_size)
    outputs: (batch_size, vector_size)
    """
    return np.mean(np.square(targets - outputs))


class Metrics:
    name2metric = {
        "CER": cal_cer,
        "WER": cal_wer,
        "BLEU": cal_bleu,
        "Cosine_Similarity": cal_cosine_similarity,
        "Accuracy": cal_accuracy,
        "Precision": cal_precision,
        "Recall": cal_recall,
        "F1": cal_f1,
        "MultiLabel_Accuracy": cal_multilabel_accuracy,
        "MultiLabel_Precision": cal_multilabel_precision,
        "MultiLabel_Recall": cal_multilabel_recall,
        "MultiLabel_F1": cal_multilabel_f1,
        "MAE": cal_mae,
        "MSE": cal_mse
    }
    def __init__(self, metrics, names=None , **metrics_arguments):
        if not isinstance(metrics, list):
            metrics = [metrics]
        self.metrics = {metric: self.name2metric[metric] for metric in metrics} if names is None else \
                       {name: self.name2metric[metric] for name, metric in zip(names, metrics)}
        self.metrics_arguments = metrics_arguments

    def __call__(self, outputs, targets=None):
        return {name: metric_func(outputs, targets, **self.metrics_arguments) for name, metric_func in self.metrics.items()}
