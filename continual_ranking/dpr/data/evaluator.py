from typing import Dict

import torch
from torch import Tensor

from continual_ranking.dpr.data import TokenizedTrainingSample


class Evaluator:

    def __init__(self):
        self.index_dataset = None
        self.k = [5] + list(range(10, 110, 10))
        self.top_k_docs = {k: 0 for k in self.k}

    def calculate_k_docs(self, scores: Tensor, batch: TokenizedTrainingSample) -> None:
        for k in self.k:
            for i, sample in enumerate(scores):
                top_items = torch.topk(scores[i], k).indices
                positive = batch.context_ids[i][0]

                for j in top_items:
                    if torch.equal(self.index_dataset[j].input_ids, positive.to('cpu')):
                        self.top_k_docs[k] += 1

    def calculate_acc(self, test_length: int) -> Dict[str, float]:
        for key in self.top_k_docs:
            self.top_k_docs[key] = self.top_k_docs[key] / test_length

        return {f'k_acc_{key}': value for key, value in self.top_k_docs.items()}
