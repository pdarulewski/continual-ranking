import logging
from collections import namedtuple
from typing import List, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from continual_ranking.dpr.data.tokenizer import Tokenizer, SimpleTokenizer

IndexSample = namedtuple(
    'IndexSample', [
        'ctxs',
    ]
)

TokenizedIndexSample = namedtuple(
    'TokenizedIndexSample', [
        'input_ids',
        'token_type_ids',
        'attention_mask',
    ]
)

logger = logging.getLogger(__name__)


class IndexTokenizer:
    def __init__(self, max_length: int):
        self.tokenizer = Tokenizer(max_length)

    def __call__(self, sample: IndexSample) -> TokenizedIndexSample:
        index_tokens = self.tokenizer(sample.ctxs)

        return TokenizedIndexSample(
            index_tokens['input_ids'].view(-1),
            index_tokens['token_type_ids'].view(-1),
            index_tokens['attention_mask'].view(-1)
        )


class IndexDataset(Dataset):

    def __init__(self, data: List[dict], tokenizer: Union[IndexTokenizer, SimpleTokenizer]):
        self.data = np.array(data)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Union[TokenizedIndexSample, List[torch.Tensor]]:
        idx = idx.to('cpu')
        try:
            return self._get_single(idx)
        except IndexError:
            return self._get_multiple(idx)

    def _get_single(self, idx: int) -> TokenizedIndexSample:
        json_sample = self.data[idx]

        sample = IndexSample(
            json_sample['ctxs'],
        )

        sample = self.tokenizer(sample)

        return sample

    def _get_multiple(self, idx: torch.Tensor) -> List[torch.Tensor]:
        data = self.data[idx]
        data = self.tokenizer([d['ctxs'] for d in data])
        return data
