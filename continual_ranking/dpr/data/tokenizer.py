from typing import List, Union

from transformers import BertTokenizer


class Tokenizer:
    def __init__(self, max_length: int):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.max_length = max_length
        self.pad_to_max = True

    def __call__(self, text: Union[str, List[str]]):
        tokens = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return tokens
