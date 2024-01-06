#coding=utf8

class Evaluator():

    def acc(self, predictions, labels):
        metric_dicts = {}
        metric_dicts['acc'] = self.accuracy(predictions, labels)
        metric_dicts['fscore'] = self.fscore(predictions, labels)
        return metric_dicts

    @staticmethod
    def accuracy(predictions, labels):
        corr, total = 0, 0
        for i, pred in enumerate(predictions):
            total += 1
            corr += set(pred) == set(labels[i])
        return 100 * corr / total

    @staticmethod
    def fscore(predictions, labels):
        TP, TP_FP, TP_FN = 0, 0, 0
        for i in range(len(predictions)):
            pred = set(predictions[i])
            label = set(labels[i])
            TP += len(pred & label)
            TP_FP += len(pred)
            TP_FN += len(label)
        if TP_FP == 0:
            precision = 0
        else:
            precision = TP / TP_FP
        recall = TP / TP_FN
        if precision + recall == 0:
            fscore = 0
        else:
            fscore = 2 * precision * recall / (precision + recall)
        return {'precision': 100 * precision, 'recall': 100 * recall, 'fscore': 100 * fscore}


from typing import Sequence


class Evaluator_bert:
    def __init__(self):
        self._n_sentences = 0
        self._n_correct_sentences = 0
        self._n_prediction_tags = 0
        self._n_truth_tags = 0
        self._n_correct_tags = 0

    def add_result(self, pred: Sequence, truth: Sequence) -> None:
        pred = set([tuple(i) for i in pred])
        truth = set([tuple(i) for i in truth])
        self._n_sentences += 1
        if pred == truth:
            self._n_correct_sentences += 1
        self._n_prediction_tags += len(pred)
        self._n_truth_tags += len(truth)
        self._n_correct_tags += len(pred & truth)

    @property
    def precision_rate(self) -> float:
        if self._n_prediction_tags == 0:
            return 0.0
        return self._n_correct_tags / self._n_prediction_tags

    @property
    def recall_rate(self) -> float:
        if self._n_truth_tags == 0:
            return 0.0
        return self._n_correct_tags / self._n_truth_tags

    @property
    def accuracy_rate(self) -> float:
        return self._n_correct_sentences / self._n_sentences

    @property
    def f1_score(self) -> float:
        p = self.precision_rate
        r = self.recall_rate
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)
