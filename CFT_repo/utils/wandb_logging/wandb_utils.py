"""WandbLogger stub — wandb disabled for clean training."""
from pathlib import Path

WANDB_ARTIFACT_PREFIX = 'wandb-artifact://'


def remove_prefix(from_string, prefix=WANDB_ARTIFACT_PREFIX):
    return from_string[len(prefix):]


def check_wandb_config_file(data_config_file):
    return None


def check_wandb_resume(opt):
    return None


def WandbLogger(opt, name, run_id, data_dict):
    return _DummyWandbLogger(data_dict)


class _DummyWandbLogger:
    """Drop-in replacement for WandbLogger with all methods as no-ops."""

    def __init__(self, data_dict):
        self.wandb = None
        self.wandb_run = None
        self.data_dict = data_dict
        self.current_epoch = 0
        self.log_imgs = 0
        self.bbox_interval = 1

    def log(self, *args, **kwargs): pass
    def log_model(self, *args, **kwargs): pass
    def log_training_progress(self, *args, **kwargs): pass
    def end_epoch(self, *args, **kwargs): pass
    def finish_run(self, *args, **kwargs): pass
