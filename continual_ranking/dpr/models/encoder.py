import logging

from torch import Tensor
from transformers import BertConfig, BertModel

logger = logging.getLogger(__name__)


class Encoder(BertModel):
    def __init__(self, config):
        BertModel.__init__(self, config)
        self.init_weights()

    @classmethod
    def init_encoder(cls):
        cfg = BertConfig.from_pretrained('bert-base-uncased')
        return cls.from_pretrained('bert-base-uncased', config=cfg)

    def forward(
            self,
            input_ids: Tensor,
            token_type_ids: Tensor,
            attention_mask: Tensor,
    ) -> Tensor:
        out = super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        return out.last_hidden_state
