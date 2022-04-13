import logging
import os
import random
import socket

import numpy as np
import torch
from omegaconf import DictConfig

logger = logging.getLogger()


def set_cfg_params_from_state(state: dict, cfg: DictConfig):
    """
    Overrides some of the encoder config parameters from a give state object
    """
    if not state:
        return

    cfg.do_lower_case = state["do_lower_case"]

    if "encoder" in state:
        saved_encoder_params = state["encoder"]
        # TODO: try to understand why cfg.encoder = state["encoder"] doesn't work

        for k, v in saved_encoder_params.items():

            # TODO: tmp fix
            if k == "q_wav2vec_model_cfg":
                k = "q_encoder_model_cfg"
            if k == "q_wav2vec_cp_file":
                k = "q_encoder_cp_file"
            if k == "q_wav2vec_cp_file":
                k = "q_encoder_cp_file"

            setattr(cfg.encoder, k, v)
    else:  # 'old' checkpoints backward compatibility support
        pass
        # cfg.encoder.pretrained_model_cfg = state["pretrained_model_cfg"]
        # cfg.encoder.encoder_model_type = state["encoder_model_type"]
        # cfg.encoder.pretrained_file = state["pretrained_file"]
        # cfg.encoder.projection_dim = state["projection_dim"]
        # cfg.encoder.sequence_length = state["sequence_length"]


def get_encoder_params_state_from_cfg(cfg: DictConfig):
    """
    Selects the param values to be saved in a checkpoint, so that a trained model can be used for downstream
    tasks without the need to specify these parameter again
    :return: Dict of params to memorize in a checkpoint
    """
    return {
        "do_lower_case": cfg.do_lower_case,
        "encoder":       cfg.encoder,
    }


def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def setup_cfg_gpu(cfg):
    """
    Setup params for CUDA, GPU & distributed training
    """
    logger.info("CFG's local_rank=%s", cfg.local_rank)
    ws = os.environ.get("WORLD_SIZE")
    cfg.distributed_world_size = int(ws) if ws else 1
    logger.info("Env WORLD_SIZE=%s", ws)

    device = str(torch.device("cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu"))
    cfg.n_gpu = torch.cuda.device_count()

    cfg.device = device

    logger.info(
        "Initialized host %s as d.rank %d on device=%s, n_gpu=%d, world size=%d",
        socket.gethostname(),
        cfg.local_rank,
        cfg.device,
        cfg.n_gpu,
        cfg.distributed_world_size,
    )
    logger.info("16-bits training: %s ", cfg.fp16)
    return cfg


def setup_logger(logger):
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    log_formatter = logging.Formatter("[%(thread)s] %(asctime)s [%(levelname)s] %(name)s: %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(log_formatter)
    logger.addHandler(console)
