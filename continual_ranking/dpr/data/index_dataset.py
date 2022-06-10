import gzip
import logging
from collections import namedtuple

from torch.utils.data import Dataset

from continual_ranking.dpr.data.tokenizer import Tokenizer

IndexSample = namedtuple(
    'IndexSample', [
        'positive_passages',
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
        index_tokens = self.tokenizer(sample.positive_passages)

        return TokenizedIndexSample(
            index_tokens['input_ids'].view(-1),
            index_tokens['token_type_ids'].view(-1),
            index_tokens['attention_mask'].view(-1)
        )


class IndexDataset(Dataset):

    def __init__(self, file_name: str, tokenizer: IndexTokenizer):
        self.tokenizer = tokenizer

        self._file_name = file_name
        # takes too long, hardcoded length
        self._length = 4_040_000

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx) -> TokenizedIndexSample:
        with gzip.open(self._file_name, 'rt') as f:
            f.seek(idx + 1)
            line = f.readline()

        sample = IndexSample(line)
        sample = self.tokenizer(sample)

        return sample
