from typing import Dict

import numpy as np
import torch
import tqdm

from continual_ranking.dpr.data.data_module import DataModule
from continual_ranking.dpr.data.file_handler import pickle_load
from continual_ranking.dpr.models.biencoder import dot_product


class Evaluator:

    def __init__(self, index_dataset, index_path: str, test_dataset, test_path: str, device: str):
        self.index_dataset = index_dataset
        self.test_dataset = test_dataset

        self.index_path = index_path
        self.test_path = test_path

        self.k = [5, 10, 20, 50, 100]
        self.top_k_docs = {k: 0 for k in self.k}

        self.device = device

    def _calculate_k_docs(self) -> None:
        test = pickle_load(self.test_path).to(self.device)
        index = pickle_load(self.index_path).to(self.device)

        self.scores = dot_product(test, index)

        self._k_docs()

    def _k_docs(self):
        index_data = [i for i in self.index_dataset]
        for k in self.k:
            top_items = torch.topk(self.scores, k).indices
            top_items = top_items.numpy()

            for i, sample in enumerate(tqdm.tqdm(top_items)):
                positive = self.test_dataset[i].context_ids[0]

                for j in sample:
                    if torch.equal(self.index_dataset[j].input_ids, positive):
                        self.top_k_docs[k] += 1

    def _calculate_acc(self) -> Dict[str, float]:
        return {f'k_acc_{key}': value / len(self.test_dataset) for key, value in self.top_k_docs.items()}

    @staticmethod
    def average_precision(actual, predicted, k=10):
        if len(predicted) > k:
            predicted = predicted[:k]

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        if not actual:
            return 0.0

        return score / min(len(actual), k)

    def mean_average_precision(self, actual, predicted, k=10):
        return np.mean([self.average_precision(a, p, k) for a, p in zip(actual, predicted)])

    def evaluate(self) -> Dict[str, float]:
        self._calculate_k_docs()
        k_scores = self._calculate_acc()
        return k_scores
