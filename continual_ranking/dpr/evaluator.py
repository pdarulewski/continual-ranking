from typing import Dict

import torch
from transformers import BertTokenizer

from continual_ranking.dpr.data.file_handler import pickle_load
from continual_ranking.dpr.models.biencoder import dot_product


class Evaluator:

    def __init__(
            self,
            max_length: int,
            index_dataset,
            index_path: str,
            test_dataset,
            test_path: str,
            device: str,
            experiment_id: int
    ):
        self.max_length = max_length

        self.index_dataset = index_dataset
        self.test_dataset = test_dataset

        self.index_path = index_path
        self.test_path = test_path

        self.k = [1, 5] + list(range(10, 110, 10))
        self.top_k_docs = {k: 0 for k in self.k}
        self.mean_ap = {k: 0 for k in self.k}

        self.device = device

        self.experiment_id = experiment_id

    def _calculate_k_docs(self) -> None:
        test = pickle_load(self.test_path).to(self.device)
        index = pickle_load(self.index_path).to(self.device)

        self.scores = dot_product(test, index)

        self._k_docs()

    def _k_docs(self) -> None:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        questions = [i['positive_ctxs'][0] for i in self.test_dataset.data]

        questions = tokenizer(
            questions,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=False,
            return_token_type_ids=False,
        ).input_ids

        answers = [i['positive_ctxs'] for i in self.index_dataset.data]

        answers = tokenizer(
            answers,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=False,
            return_token_type_ids=False
        ).input_ids

        for k in self.top_k_docs:
            top_items = torch.topk(self.scores, k)
            top_indices = top_items.indices

            for i, row in enumerate(top_indices):
                results = torch.tensor(questions[i] == answers[row])

                for j, b in enumerate(results):
                    if b.all():
                        self.top_k_docs[k] += 1
                        self.mean_ap[k] += 1 / j

    def _calculate_acc(self) -> Dict[str, float]:
        return {f'k_acc/{key}': value / len(self.test_dataset) for key, value in self.top_k_docs.items()}

    def _calculate_map(self) -> Dict[str, float]:
        return {f'k_map/{key}': value / len(self.test_dataset) for key, value in self.mean_ap.items()}

    def evaluate(self) -> Dict[str, float]:
        self._calculate_k_docs()
        k_scores = self._calculate_acc()
        map_scores = self._calculate_map()

        return {**k_scores, **map_scores, 'experiment_id': self.experiment_id}
