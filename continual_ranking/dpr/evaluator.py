import glob
import logging
from typing import Dict

import torch
from tqdm import tqdm

from continual_ranking.dpr.data.file_handler import pickle_load
from continual_ranking.dpr.models.biencoder import dot_product

logger = logging.getLogger(__name__)


class Evaluator:

    def __init__(
            self,
            index_answers,
            test_answers,
            test_path: str,
            device: str,
            experiment_id: int
    ):
        self.index_answers = index_answers
        self.test_answers = test_answers

        self.test_path = test_path

        self._max_k = 50
        self.k = [1, 5] + list(range(10, self._max_k + 10, 10))
        self.top_k_docs = {k: 0 for k in self.k}
        self.mean_ap = {k: 0 for k in self.k}

        self.device = device

        self.experiment_id = experiment_id

    @torch.no_grad()
    def _k_docs(self) -> None:
        test_encoded = pickle_load(self.test_path).to(self.device)

        top_k_all_values = []
        top_k_all_indices = []

        for index_file in glob.glob('*.index*'):
            index_encoded = pickle_load(index_file).to(self.device)
            scores = dot_product(test_encoded, index_encoded)

            top_k = torch.topk(scores, self._max_k)
            top_k_all_values.append(top_k.values)
            top_k_all_indices.append(top_k.indices)

            logger.info(f'Processed file: {index_file}')

        top_k_all_values = torch.cat([t for t in top_k_all_values], dim=1).to(self.device)
        top_k_all_indices = torch.cat([t for t in top_k_all_indices], dim=1).to(self.device)

        logger.info(f'Big top-k shape: {top_k_all_values.shape}')

        for k in self.top_k_docs:
            logger.info(f'Calculating k: {k}')

            top_k = torch.topk(top_k_all_values, k)
            top_k = torch.gather(top_k_all_indices, 1, top_k.indices)

            logger.info(f'Top-k shape: {top_k.shape}')

            for i, row in tqdm(enumerate(top_k)):
                results = self.test_answers[i] == self.index_answers[row]

                for j, b in enumerate(results):
                    if b.all():
                        self.top_k_docs[k] += 1
                        self.mean_ap[k] += 1 / (j + 1)

    def _calculate_acc(self) -> Dict[str, float]:
        return {f'k_acc/{key}': value / len(self.test_answers) for key, value in self.top_k_docs.items()}

    def _calculate_map(self) -> Dict[str, float]:
        return {f'k_map/{key}': value / len(self.test_answers) for key, value in self.mean_ap.items()}

    def evaluate(self) -> Dict[str, float]:
        self._k_docs()
        k_scores = self._calculate_acc()
        map_scores = self._calculate_map()

        return {**k_scores, **map_scores, 'experiment_id': self.experiment_id}
