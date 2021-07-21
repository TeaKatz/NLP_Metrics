import fastwer
import numpy as np

from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from .multilabel_metrics import cal_multilabel_accuracy, cal_multilabel_precision, cal_multilabel_recall, cal_multilabel_f1


def cal_cer(targets, outputs, **kwargs):
    """
    targets: List(String)
    outputs: List(String)
    """
    return fastwer.score(targets, outputs, char_level=True)


def cal_wer(targets, outputs, **kwargs):
    """
    targets: List(String)
    outputs: List(String)
    """
    return fastwer.score(targets, outputs)


def cal_bleu(targets, outputs, **lwargs):
    """
    targets: List(List(String))
    outputs: List(List(String))
    """
    bleu_score = np.mean([sentence_bleu([tar_tokens], out_tokens) for tar_tokens, out_tokens in zip(targets, outputs)])
    return bleu_score


def cal_cosine_similarity(targets, outputs, **kwargs):
    """
    target: List(Vector)
    outputs: List(Vector)
    """
    return cosine_similarity(targets, outputs)


def cal_accuracy(targets, outputs, **kwargs):
    """
    targets: List(List(Int))
    outputs: List(List(Int))
    """
    return accuracy_score(targets, outputs)


def cal_precision(targets, outputs, labels=None, pos_label=1, average="binary", sample_weight=None, zero_division="warn", **kwargs):
    """
    targets: List(List(Int))
    outputs: List(List(Int))
    """
    return precision_score(targets, outputs, labels=labels, average=average, zero_division=zero_division)


def cal_recall(targets, outputs, labels=None, pos_label=1, average="binary", sample_weight=None, zero_division="warn", **kwargs):
    """
    targets: List(List(Int))
    outputs: List(List(Int))
    """
    return recall_score(targets, outputs, labels=labels, average=average)


def cal_f1(targets, outputs, labels=None, pos_label=1, average="binary", sample_weight=None, zero_division="warn", **kwargs):
    """
    targets: List(List(Int))
    outputs: List(List(Int))
    """
    return f1_score(targets, outputs, labels=labels, average=average)


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
        "MultiLabel_F1": cal_multilabel_f1
    }
    def __init__(self, metrics, names=None , **metrics_arguments):
        if not isinstance(metrics, list):
            metrics = [metrics]
        self.metrics = {metric: self.name2metric[metric] for metric in metrics} if names is None else \
                       {name: self.name2metric[metric] for name, metric in zip(names, metrics)}
        self.metrics_arguments = metrics_arguments

    def __call__(self, outputs, targets):
        """
        outputs: list of predicted texts
        targets: list of target texts
        """
        return {name: metric(targets, outputs, **self.metrics_arguments) for name, metric in self.metrics.items()}