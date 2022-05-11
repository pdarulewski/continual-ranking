from typing import Dict

import numba
import numba.cuda
import torch

from continual_ranking.dpr.data.data_module import DataModule
from continual_ranking.dpr.data.file_handler import pickle_load
from continual_ranking.dpr.models.biencoder import dot_product


@numba.jit(parallel=True)
def _k_docs(scores, index_dataset, test_dataset):
    full_k = [5] + list(range(10, 110, 10))
    top_k_docs = {k: 0 for k in full_k}

    for i in numba.prange(scores):
        for k in full_k:
            top_items = torch.topk(scores[i], k).indices
            positive = test_dataset[i].context_ids[0]
            for j in top_items:
                if torch.equal(index_dataset[j].input_ids, positive):
                    top_k_docs[k] += 1

        if i % 100 == 0:
            print(i)

    return top_k_docs


class Evaluator:

    def __init__(self, index_dataset, index_path: str, test_dataset, test_path: str, device: str):
        self.index_dataset = index_dataset
        self.test_dataset = test_dataset

        self.index_path = index_path
        self.test_path = test_path

        self.k = [5] + list(range(10, 110, 10))
        self.top_k_docs = {k: 0 for k in self.k}

        self.device = device

    def _calculate_k_docs(self) -> None:
        test = pickle_load(self.test_path).to(self.device)
        index = pickle_load(self.index_path).to(self.device)

        scores = dot_product(test, index).numpy()

        self.top_k_docs = _k_docs(scores, self.index_dataset, self.test_dataset)

    def _calculate_acc(self) -> Dict[str, float]:
        return {f'k_acc_{key}': value / len(self.test_dataset) for key, value in self.top_k_docs.items()}

    def evaluate(self) -> Dict[str, float]:
        self._calculate_k_docs()
        k_scores = self._calculate_acc()
        return k_scores
