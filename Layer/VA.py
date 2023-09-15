import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F

class VA(nn.Module):
    """Parent class for leaky integrate and fire neuron models."""
    def __init__(
        self,
        in_feature,
        out_af=3,
    ):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = in_feature
        self.out_af = out_af

        self.wa_x = Parameter(torch.empty((self.out_af, self.in_feature)))
        self.ba = Parameter(torch.empty(self.out_af))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.out_feature) if self.out_feature > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x):
        batch, _ = x.size()
        x = x.unsqueeze(2)

        wa_x = self.wa_x.unsqueeze(0).tile((batch, 1, 1))
        ba = self.ba.unsqueeze(1)

        wa = torch.sigmoid(torch.bmm(wa_x, x) + ba)

        h = torch.cat((torch.tanh(x), F.softplus(x), torch.relu(x)), dim=2)
        y = torch.tanh(torch.bmm(h, wa))

        return y.squeeze(2)
