import logging
from collections import namedtuple
from typing import List

from torch.utils.data import Dataset

from continual_ranking.dpr.data.tokenizer import Tokenizer

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

    def __init__(self, data: List[dict], tokenizer: IndexTokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> TokenizedIndexSample:
        json_sample = self.data[idx]

        sample = IndexSample(
            json_sample['ctxs'],
        )

        sample = self.tokenizer(sample)

        return sample
