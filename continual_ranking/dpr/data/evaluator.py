from typing import Dict

import torch


class Evaluator:

    def __init__(self, index_dataset, test_dataset, test):
        self.index_dataset = index_dataset
        self.test_dataset = test_dataset
        self.test = test
        self.k = [5] + list(range(10, 110, 10))

        self.top_k_docs = {k: 0 for k in self.k}

    def calculate_k_docs(self) -> Dict[str, float]:
        for k in self.k:
            for i, sample in enumerate(self.test_dataset):
                top_items = torch.topk(self.test[i], k).indices
                positive = sample.context_ids[0]

                for j in top_items:
                    if torch.equal(self.index_dataset[j].input_ids, positive):
                        self.top_k_docs[k] += 1

        for key in self.top_k_docs:
            self.top_k_docs[key] = self.top_k_docs[key] / len(self.test_dataset)

        return {f'k_acc_{key}': value for key, value in self.top_k_docs.items()}
