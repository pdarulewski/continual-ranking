import collections
import glob
import logging
import os

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.serialization import default_restore_location

logger = logging.getLogger()

CheckpointState = collections.namedtuple(
    "CheckpointState",
    [
        "model_dict",
        "optimizer_dict",
        "scheduler_dict",
        "offset",
        "epoch",
        "encoder_params",
    ],
)


def move_to_device(sample, device):
    if len(sample) == 0:
        return {}

    def _move_to_device(maybe_tensor, device):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.to(device)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_device(value, device) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list) or isinstance(maybe_tensor, tuple):
            return [_move_to_device(x, device) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_device(sample, device)


def get_schedule_linear(
        optimizer,
        warmup_steps,
        total_training_steps,
        steps_shift=0,
        last_epoch=-1,
):
    """Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        current_step += steps_shift
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            1e-7,
            float(total_training_steps - current_step) / float(max(1, total_training_steps - warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_model_obj(model: nn.Module):
    return model.module if hasattr(model, "module") else model


def get_model_file(args, file_prefix) -> str:
    if args.model_file and os.path.exists(args.model_file):
        return args.model_file

    out_cp_files = glob.glob(os.path.join(args.output_dir, file_prefix + "*")) if args.output_dir else []
    logger.info("Checkpoint files %s", out_cp_files)
    model_file = None

    if len(out_cp_files) > 0:
        model_file = max(out_cp_files, key=os.path.getctime)
    return model_file


def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    logger.info("Reading saved model from %s", model_file)
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, "cpu"))
    logger.info("model_state_dict keys %s", state_dict.keys())
    return CheckpointState(**state_dict)
