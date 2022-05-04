import torch
from torch import Tensor
from transformers import BertTokenizer


class Tensorizer:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.max_length = 256
        self.pad_to_max = True

    def text_to_tensor(self, text: str, title: str = None):
        text = text.strip()

        if title:
            segment_1 = title
            segment_2 = text
        else:
            segment_1 = text
            segment_2 = None

        token_ids = self.tokenizer.encode(
            segment_1,
            text_pair=segment_2,
            add_special_tokens=True,
            max_length=self.max_length,
            pad_to_max_length=False,
            truncation=True,
        )

        seq_len = self.max_length
        if self.pad_to_max and len(token_ids) < seq_len:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (seq_len - len(token_ids))
        if len(token_ids) >= seq_len:
            token_ids = token_ids[0:seq_len]
            token_ids[-1] = self.tokenizer.sep_token_id

        return torch.tensor(token_ids)

    def get_attn_mask(self, tokens_tensor: Tensor) -> Tensor:
        return tokens_tensor != self.tokenizer.pad_token_id

    @staticmethod
    def pad_to_len(seq: Tensor, pad_id: int, max_len: int):
        s_len = seq.size(0)
        if s_len > max_len:
            return seq[0:max_len]
        return torch.cat([seq, torch.Tensor().new_full((max_len - s_len,), pad_id, dtype=torch.long)], dim=0)
