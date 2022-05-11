from typing import Dict

import torch
from torch import Tensor

from continual_ranking.dpr.data.file_handler import pickle_load
from continual_ranking.dpr.models.biencoder import dot_product


class Evaluator:

    def __init__(self, index_dataset, index_path: str, test_dataset, test_path: str):
        self.index_dataset = index_dataset
        self.test_dataset = test_dataset

        self.index_path = index_path
        self.test_path = test_path

        self.k = [5] + list(range(10, 110, 10))
        self.top_k_docs = {k: 0 for k in self.k}

    def _index_test_dot_product(self):
        index = pickle_load(self.index_path)
        test = pickle_load(self.test_path)
        scores = dot_product(index, test)
        return scores

    def _calculate_k_docs(self, scores: Tensor) -> None:
        for i, sample in enumerate(scores):
            for k in self.k:
                top_items = torch.topk(scores[i], k).indices
                # FIXME: process that in batches
                positive = self.test_dataset[i * 16].context_ids[i % 16][0]

                for j in top_items:
                    if torch.equal(self.index_dataset[j].input_ids, positive.to('cpu')):
                        self.top_k_docs[k] += 1

    def _calculate_acc(self) -> Dict[str, float]:
        for key in self.top_k_docs:
            self.top_k_docs[key] = self.top_k_docs[key] / len(self.test_dataset)

        return {f'k_acc_{key}': value for key, value in self.top_k_docs.items()}

    def evaluate(self) -> Dict[str, float]:
        scores = self._index_test_dot_product()
        self._calculate_k_docs(scores)
        k_scores = self._calculate_acc()
        return k_scores
