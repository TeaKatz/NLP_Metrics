from ..Metrics import Metrics


class STSBMetric:
    def __init__(self):
        self.metrics = Metrics(["Spearman_Correlation", "MAE"])

    def __call__(self, preds, targs):
        """
        preds: (batch_size, )
        targs: (batch_size, )
        """
        return self.metrics(preds, targs)