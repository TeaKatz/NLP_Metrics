import numpy as np
from ..Metrics import Metrics


class WNLIMetric:
    def __init__(self):
        self.metrics = Metrics(["Accuracy"])

    def __call__(self, preds, targs):
        """
        preds: (batch_size, vector_size)
        targs: (batch_size, )
        """
        # (batch_size, )
        preds = np.argmax(preds, axis=-1)
        return self.metrics(preds, targs)