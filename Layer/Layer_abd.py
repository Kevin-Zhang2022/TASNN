import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F

class A_Snn(nn.Module):
    """Parent class for leaky integrate and fire neuron models."""
    def __init__(
        self,
        in_feature,
        out_feature,
        mode,
        out_af=3,
    ):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.mode = mode
        self.out_af = out_af
        if mode == 'lstm':
            self.wi_x = Parameter(torch.empty((self.out_feature, self.in_feature)))
            self.wi_h = Parameter(torch.empty(self.out_feature, self.out_feature))
            self.bi = Parameter(torch.empty(self.out_feature))

            self.wf_x = Parameter(torch.empty((self.out_feature, self.in_feature)))
            self.wf_h = Parameter(torch.empty(self.out_feature, self.out_feature))
            self.bf = Parameter(torch.empty(self.out_feature))

            self.wg_x = Parameter(torch.empty((self.out_feature, self.in_feature)))
            self.wg_h = Parameter(torch.empty(self.out_feature, self.out_feature))
            self.bg = Parameter(torch.empty(self.out_feature))

            self.wo_x = Parameter(torch.empty((self.out_feature, self.in_feature)))
            self.wo_h = Parameter(torch.empty(self.out_feature, self.out_feature))
            self.bo = Parameter(torch.empty(self.out_feature))
        elif mode== 'asnn':
            self.ww_x = Parameter(torch.empty((self.out_feature, self.in_feature)))
            self.ww_y = Parameter(torch.empty(self.out_feature, self.out_feature))
            self.bw = Parameter(torch.empty(self.out_feature))

            self.wb_x = Parameter(torch.empty((self.out_feature, self.in_feature)))
            self.wb_y = Parameter(torch.empty(self.out_feature, self.out_feature))
            self.bb = Parameter(torch.empty(self.out_feature))

            self.wa_x = Parameter(torch.empty((self.out_af, self.in_feature)))
            self.wa_y = Parameter(torch.empty(self.out_af, self.out_feature))
            self.ba = Parameter(torch.empty(self.out_af))
            # self.wg_x = Parameter(torch.empty((self.out_feature, self.in_feature)))
            # self.wg_h = Parameter(torch.empty(self.out_feature, self.out_feature))
            # self.bg = Parameter(torch.empty(self.out_feature))
            #
            self.wo_x = Parameter(torch.empty((self.out_feature, self.in_feature)))
            self.wo_y = Parameter(torch.empty(self.out_feature, self.out_feature))
            self.bo = Parameter(torch.empty(self.out_feature))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.out_feature) if self.out_feature > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x):
        batch, T, _ = x.size()
        out = torch.zeros(batch, T, self.out_feature)
        # y = torch.zeros(batch, self.out_feature, 1)
        if self.mode=='lstm':
            wi_x = self.wi_x.unsqueeze(0).tile((batch, 1, 1))
            wi_h = self.wi_h.unsqueeze(0).tile((batch, 1, 1))
            bi = self.bi.unsqueeze(1)

            wf_x = self.wf_x.unsqueeze(0).tile((batch, 1, 1))
            wf_h = self.wf_h.unsqueeze(0).tile((batch, 1, 1))
            bf = self.bf.unsqueeze(1)

            wg_x = self.wg_x.unsqueeze(0).tile((batch, 1, 1))
            wg_h = self.wg_h.unsqueeze(0).tile((batch, 1, 1))
            bg = self.bg.unsqueeze(1)

            wo_x = self.wo_x.unsqueeze(0).tile((batch, 1, 1))
            wo_h = self.wo_h.unsqueeze(0).tile((batch, 1, 1))
            bo = self.bo.unsqueeze(1)

            c = torch.zeros(batch, self.out_feature, 1)
            h = torch.zeros(batch, self.out_feature, 1)

            # wx_l = self.wx_l.unsqueeze(0).tile((batch, 1, 1))
            for t in range(T):
                xt = x[:,t,:].unsqueeze(2)
                i = torch.sigmoid(torch.bmm(wi_x, xt) + torch.bmm(wi_h, h) + bi) # i
                f = torch.sigmoid(torch.bmm(wf_x, xt) + torch.bmm(wf_h, h) + bf) # f
                g = torch.tanh(torch.bmm(wg_x, xt) + torch.bmm(wg_h, h) + bg)
                o = torch.sigmoid(torch.bmm(wo_x, xt) + torch.bmm(wo_h, h) + bo)

                # wx = torch.sigmoid(torch.bmm(wx_x, xt) + torch.bmm(wx_y, y) + bx)
                c = c*f+i*g
                h = o*torch.tanh(c)
                # y = h
                # # wx_l = torch.sigmoid(torch.bmm(wx_x, xt) + torch.bmm(wx_y, y) + bx)
                # # mask = y > b
                # # y = w*y + wx*torch.bmm(wx_l,xt) + b
                # y = w * y + b*
                # h =
                out[:, t, :] = h.squeeze(2)
            # plt.plot(out.detach().numpy()[0,:,0])
            return out, h.squeeze(2)
        elif self.mode=='asnn':
            ww_x = self.ww_x.unsqueeze(0).tile((batch, 1, 1))
            ww_y = self.ww_y.unsqueeze(0).tile((batch, 1, 1))
            bw = self.bw.unsqueeze(1)

            wb_x = self.wb_x.unsqueeze(0).tile((batch, 1, 1))
            wb_y = self.wb_y.unsqueeze(0).tile((batch, 1, 1))
            bb = self.bb.unsqueeze(1)

            wa_x = self.wa_x.unsqueeze(0).tile((batch, 1, 1))
            wa_y = self.wa_y.unsqueeze(0).tile((batch, 1, 1))
            ba = self.ba.unsqueeze(1)

            # wo_x = self.wo_x.unsqueeze(0).tile((batch, 1, 1))
            # wo_y = self.wo_y.unsqueeze(0).tile((batch, 1, 1))
            # bo = self.bo.unsqueeze(1)


            # c = torch.zeros(batch, self.out_feature, 1)
            y = torch.zeros(batch, self.out_feature, 1)
            h = torch.zeros(batch, self.out_feature, 1)

            # wx_l = self.wx_l.unsqueeze(0).tile((batch, 1, 1))
            for t in range(T):
                xt = x[:,t,:].unsqueeze(2)
                w = torch.sigmoid(torch.bmm(ww_x, xt) + torch.bmm(ww_y, h) + bw)  # i
                b = torch.tanh(torch.bmm(wb_x, xt) + torch.bmm(wb_y, h) + bb)  # f
                wa = torch.sigmoid(torch.bmm(wa_x, xt) + torch.bmm(wa_y, h) + ba)
                # wo = torch.sigmoid(torch.bmm(wo_x, xt) + torch.bmm(wo_y, h) + bo)

                y = w*y+b

                # h = torch.cat((torch.tanh(y), F.softplus(y), torch.relu(y)), dim=2)
                h = torch.cat((F.leaky_relu_(y), torch.tanh(y), F.softplus(y)), dim=2)
                h = torch.tanh(torch.bmm(h, wa)/10)

                # for i in range(20):
                #     plt.plot(wa[i].detach().numpy())

                out[:, t, : ] = h.squeeze(2)
            return out, y.squeeze(2)
