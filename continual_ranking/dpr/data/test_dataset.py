from collections import namedtuple
from typing import List

from torch.utils.data import Dataset

from continual_ranking.dpr.data.tokenizer import Tokenizer

TestSample = namedtuple(
    'TestSample', [
        'query',
        'positive_passages',
    ]
)

TokenizedTestSample = namedtuple(
    'TokenizedTestSample', [
        'input_ids',
        'token_type_ids',
        'attention_mask',
    ]
)


class TestTokenizer:
    def __init__(self, max_length: int):
        self.tokenizer = Tokenizer(max_length)

    def __call__(self, sample: TestSample):
        ctx_tokens = self.tokenizer(sample.positive_passages)

        return TokenizedTestSample(
            ctx_tokens['input_ids'],
            ctx_tokens['token_type_ids'],
            ctx_tokens['attention_mask']
        )


class TestDataset(Dataset):

    def __init__(self, data: List[dict], tokenizer: TestTokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        json_sample = self.data[idx]

        sample = TestSample(
            json_sample['question'],
        )

        sample = self.tokenizer(sample)

        return sample



