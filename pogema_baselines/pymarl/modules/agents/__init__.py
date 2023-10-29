REGISTRY = {}

from .rnn_agent import RNNAgent
from .dqn_agent import DQNAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["dqn"] = DQNAgent