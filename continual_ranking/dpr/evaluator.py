import glob
from typing import Dict

import torch
from transformers import BertTokenizer

from continual_ranking.dpr.data.file_handler import pickle_load
from continual_ranking.dpr.data.index_dataset import IndexDataset
from continual_ranking.dpr.data.train_dataset import TrainDataset
from continual_ranking.dpr.models.biencoder import dot_product


class Evaluator:

    def __init__(
            self,
            max_length: int,
            index_dataset,
            test_dataset,
            test_path: str,
            device: str,
            experiment_id: int
    ):
        self.max_length = max_length

        self.index_dataset: IndexDataset = index_dataset
        self.test_dataset: TrainDataset = test_dataset

        self.test_path = test_path

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self._max_k = 50

        self.k = [1, 5] + list(range(10, self._max_k + 10, 10))
        self.top_k_docs = {k: 0 for k in self.k}
        self.mean_ap = {k: 0 for k in self.k}

        self.device = device

        self.experiment_id = experiment_id

    def _tokenize_test(self) -> torch.Tensor:
        test_answers = [i['positive_ctxs'][0] for i in self.test_dataset.data]
        test_answers = self.tokenizer(
            test_answers,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=False,
            return_token_type_ids=False,
        ).input_ids

        return test_answers

    @torch.no_grad()
    def _k_docs(self) -> None:
        test_answers = self._tokenize_test()
        test_encoded = pickle_load(self.test_path).to(self.device)

        top_k_values = []
        top_k_indices = []

        for index_file in glob.glob('*.index*'):
            index_encoded = pickle_load(index_file).to(self.device)
            scores = dot_product(test_encoded, index_encoded)
            top_k = torch.topk(scores, self._max_k)

            top_k_indices.append(top_k.indices)
            top_k_values.append(top_k.values)

        top_k_values = torch.cat([t for t in top_k_values], dim=1)
        top_k_indices = torch.cat([t for t in top_k_indices], dim=1)

        top_k = torch.topk(top_k_values, self._max_k)
        top_k = torch.gather(top_k_indices, 1, top_k.indices)

        for k in self.top_k_docs:
            for i, row in enumerate(top_k):
                results = torch.tensor(test_answers[i] == self.index_dataset[row].input_ids)

                for j, b in enumerate(results):
                    if b.all():
                        self.top_k_docs[k] += 1
                        self.mean_ap[k] += 1 / (j + 1)

    def _calculate_acc(self) -> Dict[str, float]:
        return {f'k_acc/{key}': value / len(self.test_dataset) for key, value in self.top_k_docs.items()}

    def _calculate_map(self) -> Dict[str, float]:
        return {f'k_map/{key}': value / len(self.test_dataset) for key, value in self.mean_ap.items()}

    def evaluate(self) -> Dict[str, float]:
        self._k_docs()
        k_scores = self._calculate_acc()
        map_scores = self._calculate_map()

        return {**k_scores, **map_scores, 'experiment_id': self.experiment_id}
