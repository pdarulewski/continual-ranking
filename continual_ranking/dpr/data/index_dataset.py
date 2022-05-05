from collections import namedtuple
from typing import List

import torch
from torch.utils.data import Dataset

from continual_ranking.dpr.data.tokenizer import Tokenizer

IndexSample = namedtuple(
    'IndexSample', [
        'query',
        'positive_passages',
    ]
)

TokenizedIndexSample = namedtuple(
    'TokenizedIndexSample', [
        'index_ids',
        'index_segments',
        'index_attn_mask',
    ]
)


class IndexDataset(Dataset):

    def __init__(self, data: List[dict]):
        self.data = data
        self.tokenizer = IndexTokenizer()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        json_sample = self.data[idx]

        sample = IndexSample(
            json_sample['question'],
            json_sample['positive_ctxs'],
        )

        sample = self.tokenizer(sample)

        return sample


class IndexTokenizer:
    def __init__(self):
        self.tensorizer = Tokenizer()

    def __call__(self, sample: IndexSample):
        index_ids = self.tensorizer.tokenize(sample.positive_passages, sample.query)
        index_segments = torch.zeros_like(index_ids)
        index_attn_mask = self.tensorizer.get_attn_mask(index_ids)

        return TokenizedIndexSample(
            index_ids,
            index_segments,
            index_attn_mask,
        )
