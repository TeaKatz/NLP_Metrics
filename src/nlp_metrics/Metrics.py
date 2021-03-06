import fastwer
import numpy as np

from nltk.translate.bleu_score import corpus_bleu
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from .multilabel_metrics import cal_multilabel_accuracy, cal_multilabel_precision, cal_multilabel_recall, cal_multilabel_f1


def cal_cer(preds, targs, **kwargs):
    """
    targs: List(String)
    preds: List(String)
    """
    return fastwer.score(targs, preds, char_level=True)


def cal_wer(preds, targs, **kwargs):
    """
    targs: List(String)
    preds: List(String)
    """
    return fastwer.score(targs, preds)


def cal_bleu(preds, targs, **kwargs):
    """
    targs: List(List(String))
    preds: List(List(String))
    """
    bleu_score = corpus_bleu([targs], preds)
    return bleu_score


def cal_cosine_similarity(preds, targs, **kwargs):
    """
    target: (batch_size, vector_size)
    preds: (batch_size, vector_size)
    """
    return cosine_similarity(targs, preds).diagonal().mean()


def cal_accuracy(preds, targs, **kwargs):
    """
    targs: (batch_size, )
    preds: (batch_size, )
    """
    return accuracy_score(targs, preds)


def cal_precision(preds, targs, labels=None, pos_label=1, average="binary", sample_weight=None, zero_division="warn", **kwargs):
    """
    targs: (batch_size, )
    preds: (batch_size, )
    """
    return precision_score(targs, preds, labels=labels, average=average, zero_division=zero_division)


def cal_recall(preds, targs, labels=None, pos_label=1, average="binary", sample_weight=None, zero_division="warn", **kwargs):
    """
    targs: (batch_size, )
    preds: (batch_size, )
    """
    return recall_score(targs, preds, labels=labels, average=average)


def cal_f1(preds, targs, labels=None, pos_label=1, average="binary", sample_weight=None, zero_division="warn", **kwargs):
    """
    targs: (batch_size, )
    preds: (batch_size, )
    """
    return f1_score(targs, preds, labels=labels, average=average)


def cal_mae(preds, targs, **kwargs):
    """
    targs: (batch_size, vector_size)
    preds: (batch_size, vector_size)
    """
    return np.mean(np.abs(targs - preds))


def cal_mse(preds, targs, **kwargs):
    """
    targs: (batch_size, vector_size)
    preds: (batch_size, vector_size)
    """
    return np.mean(np.square(targs - preds))


def cal_spearman_correlation(preds, targs, **kwargs):
    """
    targs: (batch_size, )
    preds: (batch_size, )
    """
    assert len(preds) == len(targs)

    correlation, _ = spearmanr(preds, targs)
    return correlation


def cal_pearson_correlation(preds, targs, **kwargs):
    """
    targs: (batch_size, )
    preds: (batch_size, )
    """
    assert len(preds) == len(targs)

    correlation, _ = pearsonr(preds, targs)
    return correlation


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
        "MSE": cal_mse,
        "Spearman_Correlation": cal_spearman_correlation,
        "Pearson_Correlation": cal_pearson_correlation
    }
    def __init__(self, metrics, names=None , **metrics_arguments):
        if not isinstance(metrics, list):
            metrics = [metrics]
        self.metrics = {metric: self.name2metric[metric] for metric in metrics} if names is None else \
                       {name: self.name2metric[metric] for name, metric in zip(names, metrics)}
        self.metrics_arguments = metrics_arguments

    def __call__(self, *args):
        return {name: metric_func(*args, **self.metrics_arguments) for name, metric_func in self.metrics.items()}
