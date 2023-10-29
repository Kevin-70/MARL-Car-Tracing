from functools import partial
import sys
sys.path.append("/home/maddpg/pogema_env")
from pogema_ours import pogema_v0

from utils.config_validation import Environment, ExperimentConfig


def env_fn(**kwargs):
    cfg = Environment(**kwargs)
    cfg.grid_config.integration = 'PyMARL'
    return pogema_v0(grid_config=cfg.grid_config)

REGISTRY = {}

REGISTRY["pogema"] = partial(env_fn, integration='PyMARL')
