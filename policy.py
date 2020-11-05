import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, s_size=4, h_size=8, a_size=2):
        super(Policy, self).__init__()
        
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.selu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
    
    def act(self, state, device):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        # >>> m = Categorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
        # >>> m.sample()  # equal probability of 0, 1, 2, 3
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)