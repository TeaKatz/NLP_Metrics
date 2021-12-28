import numpy as np
from ..Metrics import Metrics


class MNLIMetric:
    def __init__(self):
        self.metrics = Metrics(["Accuracy"])

    def __call__(self, preds, targs):
        """
        preds: (batch_size, vector_size)
        targs: (batch_size, )
        """
        # (batch_size, )
        preds = np.argmax(preds)
        return self.metrics(preds, targs)