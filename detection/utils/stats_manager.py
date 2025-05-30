import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


class StatsManager:

    def __init__(self, config):
        self.config = config

    def get_stats(self, predictions, labels):
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)

        # Get probability of positive class
        positive_scores = predictions[:, 1]

        roc_auc = roc_auc_score(labels, positive_scores)
        fpr, tpr, thresholds = roc_curve(labels, positive_scores)

        # Compute EER
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.abs(fpr - fnr))]

        predictions = predictions.argmax(-1)
        accuracy = (labels == predictions).sum() / len(labels)
        return accuracy, roc_auc, eer
