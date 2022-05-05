from collections import namedtuple
from typing import List

import torch
from torch.utils.data import Dataset

from continual_ranking.dpr.data.tokenizer import Tokenizer

TestSample = namedtuple(
    'TestSample', [
        'query',
    ]
)

TokenizedTestSample = namedtuple(
    'TokenizedTestSample', [
        'question_ids',
        'question_segments',
        'question_attn_mask',
    ]
)


class TestDataset(Dataset):

    def __init__(self, data: List[dict]):
        self.data = data
        self.tokenizer = TestTokenizer()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        json_sample = self.data[idx]

        sample = TestSample(
            json_sample['question'],
        )

        sample = self.tokenizer(sample)

        return sample


class TestTokenizer:
    def __init__(self):
        self.tensorizer = Tokenizer()

    def __call__(self, sample: TestSample):
        question_ids = self.tensorizer.tokenize(sample.query)
        question_segments = torch.zeros_like(question_ids)
        question_attn_mask = self.tensorizer.get_attn_mask(question_ids)

        return TokenizedTestSample(
            question_ids,
            question_segments,
            question_attn_mask,
        )
