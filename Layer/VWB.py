import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F

class VWB(nn.Module):
    """Parent class for leaky integrate and fire neuron models."""
    def __init__(
        self,
        in_feature,
        out_feature,
    ):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.ww_x = Parameter(torch.empty((self.out_feature, self.in_feature)))
        self.bw = Parameter(torch.empty(self.out_feature))

        self.wb_x = Parameter(torch.empty((self.out_feature, self.in_feature)))
        self.bb = Parameter(torch.empty(self.out_feature))

        self.wh_x = Parameter(torch.empty((self.out_feature, self.in_feature)))
        self.bh = Parameter(torch.empty(self.out_feature))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.out_feature) if self.out_feature > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x):
        batch, _ = x.size()
        x = x.unsqueeze(2)

        ww_x = self.ww_x.unsqueeze(0).tile((batch, 1, 1))
        # ww_y = self.ww_y.unsqueeze(0).tile((batch, 1, 1))
        bw = self.bw.unsqueeze(1)

        wb_x = self.wb_x.unsqueeze(0).tile((batch, 1, 1))
        # wb_y = self.wb_y.unsqueeze(0).tile((batch, 1, 1))
        bb = self.bb.unsqueeze(1)

        wh_x = self.wh_x.unsqueeze(0).tile((batch, 1, 1))
        # wb_y = self.wb_y.unsqueeze(0).tile((batch, 1, 1))
        bh = self.bh.unsqueeze(1)

        w = torch.sigmoid(torch.bmm(ww_x, x) + bw)  # i
        b = torch.tanh(torch.bmm(wb_x, x) + bb)  # f
        h = torch.sigmoid(torch.bmm(wh_x, x) + bh)
        # wo = torch.sigmoid(torch.bmm(wo_x, xt) + torch.bmm(wo_y, h) + bo)

        y = w*h+b

        # h = torch.cat((torch.tanh(y), F.softplus(y), torch.relu(y)), dim=2)
        # h = torch.cat((torch.tanh(y), torch.tanh(y), torch.tanh(y)), dim=2)
        # h = torch.bmm(h, wa)

        return y.squeeze(2)
