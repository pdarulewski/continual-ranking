from .hf_models import get_bert_biencoder_components


def init_hf_bert_biencoder(args, **kwargs):
    return get_bert_biencoder_components(args, **kwargs)


BIENCODER_INITIALIZERS = {
    "hf_bert": init_hf_bert_biencoder,
}


def init_biencoder_components(encoder_type: str, args, **kwargs):
    return BIENCODER_INITIALIZERS[encoder_type](args, **kwargs)
