import torch.nn as nn
import torch.nn.functional as F


class DQNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(DQNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim1)
        self.fc2 = nn.Linear(args.hidden_dim1, args.hidden_dim2)
        self.fc3 = nn.Linear(args.hidden_dim2, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim1).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x, hidden_state
