from abc import ABC

import matplotlib.pyplot as plt
import numba
import snntorch
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
import snntorch as snn
import torch.nn.functional as F
from scipy import stats
import numpy as np
import pyfilterbank.gammatone as gt
from cla.GlobalParameter import GlobalParameter as gp


# from main.train import gp
# from main.train import sample_rate,band_width,channels

class Basic(nn.Module):
    def __init__(self, in_features, out_features):
        super(Basic, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mean = None
        self.scale = None
        self.of=None

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.out_features) if self.out_features > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    @classmethod
    def get_cf(cls, start, end, channels):
        start_band = gt.hertz_to_erbscale(start)
        end_band = gt.hertz_to_erbscale(end)

        density = (end_band - start_band) / (channels - 1 * 0.05)
        gtb = gt.GammatoneFilterbank(samplerate=10000, bandwidth_factor=0.05, order=4, startband=start_band,
                                     endband=end_band, density=density, normfreq=start_band)
        return gtb.centerfrequencies

    def reset_mask(self, scale_range):
        in_features = self.in_features
        out_features = self.out_features

        mean = self.get_cf(0, in_features, out_features)
        scale = self.get_cf(scale_range[0], scale_range[1], out_features)
        self.mean = mean
        self.scale = scale

        # mean = np.arange(0,in_features,in_features/out_features)
        # scale = np.arange(scale_range[0],scale_range[1],(scale_range[1]-scale_range[0])/out_features)
        x = np.arange(0, in_features, 1)
        # plt.figure()
        # plt.plot(scale)
        # scale = (1+scale)
        mask = []
        for i in range(out_features):
            norm = stats.norm(loc=mean[i], scale=scale[i])
            y = norm.pdf(x) * (np.sqrt(2 * np.pi) * scale[i])
            mask.append(y)
            # plt.plot(y)
            # plt.show()


        mask = np.array(mask)
        #
        # plt.imshow(mask,aspect='auto')
        sum=0
        for i in range(out_features-1):
            sum += np.sum(mask[i,:] * mask[i+1,:])
        self.of = sum/out_features


        mask = torch.tensor(mask, dtype=torch.float32)

        # temp = mask
        # std_x = np.std(temp,axis=1)
        # std_x = std_x.reshape((std_x.shape[0], 1))
        # std_mat = np.matmul(std_x,std_x.transpose())
        # corcoe = np.cov(temp)/(std_mat+1e-10)
        # print(np.mean(corcoe))

        # plt.imshow(mask)
        # for i in range(50):
        # plt.plot(mask[i,:])
        # # plt.plot(y)
        # # plt.imshow(self.mat.detach().numpy())
        return mask


class Cochlear(nn.Module):
    def __init__(self,
                 frequency_range,
                 channels,
                 bandwidth_factor,
                 sample_rate,
                 ):
        super(Cochlear, self).__init__()
        start_band = gt.hertz_to_erbscale(frequency_range[0])
        end_band = gt.hertz_to_erbscale(frequency_range[1])
        self.gtfb = gt.GammatoneFilterbank(samplerate=sample_rate, bandwidth_factor=bandwidth_factor,
                                           order=4, startband=start_band, endband=end_band,
                                           density=(end_band - start_band) / (channels-0.05), normfreq=0)
        self.cf = self.gtfb.centerfrequencies

    def forward(self, inp):
        out = []
        for b in range(inp.shape[0]):
            results = self.gtfb.analyze(inp[b, :])
            temp = []
            for (band, state) in results:
                temp.append(np.real(band))
            temp = np.array(temp).transpose()
            out.append(temp)
        out = np.array(out)
        return torch.tensor(out, dtype=torch.float32)
    # plt.plot(out[0,:,25])


class InnerHairCell(nn.Module):
    def __init__(self,
                 window,
                 stride,
                 mode='half_mean',
    ):
        super(InnerHairCell, self).__init__()
        self.window = window
        self.stride = stride
        self.mode=mode

    def forward(self, inp):
        if self.mode=='abs_mean':
            inp = abs(inp)
        else:
            inp = (inp > 0) * inp

        #

        # org
        # # do nothing

        out = []
        for head in range(0, inp.size(1), self.stride):
            temp = torch.mean(inp[:, head:(head + self.window), :], dim=1)
            out.append(temp.unsqueeze(1))
        out = torch.cat(out, dim=1)
        # plt.plot(inp[0,:,25])
        # max,_ = torch.max(out,dim=1)
        # plt.plot(max[15])
        # 0,1,5,15
        return out
        # plt.plot(out[0,:,25].detach().numpy())


class AuditoryNerve(nn.Module):
    def __init__(self, ans, range):
        super(AuditoryNerve, self).__init__()
        self.ans = ans
        self.range = range

    def forward(self, x):
        valve_array = torch.arange(self.range[0], self.range[1], (self.range[1] - self.range[0]) / self.ans).unsqueeze(
            0).tile((1, x.size(1)))
        inp = torch.repeat_interleave(x, dim=1, repeats=self.ans)
        out = (inp >= valve_array).float()
        return out

        # plt.imshow(out[0,:].unsqueeze(0).detach().numpy(),aspect='auto')
        pass


class Bushy(Basic):
    def __init__(
            self,
            in_features,
            out_features,
            scale_range,  # variance range
    ):
        super(Bushy, self).__init__(in_features, out_features)
        self.sleaky = snn.Leaky(beta=0.95)
        self.mask = self.reset_mask(scale_range)

    def forward(self, inp, mem):
        spk, mem = self.sleaky(torch.matmul(self.mask.squeeze(0), inp.unsqueeze(-1)).squeeze(-1), mem)
        return spk, mem


class E_Linear(Basic):
    def __init__(
            self,
            in_features,
            out_features,
    ):
        super(E_Linear, self).__init__(in_features, out_features)
        self.w = Parameter(torch.empty((out_features)))
        self.reset_parameters()

    def forward(self, inp):
        out = inp * self.w
        return out


class InferiorColliculus(Basic):
    def __init__(
            self,
            in_features,
            out_features,
    ):
        super(InferiorColliculus, self).__init__(in_features, out_features)
        self.sleaky = snn.Leaky(beta=0.95)
        self.elinear = E_Linear(in_features, out_features)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, inp, mem=None):
        if mem == None:
            out = self.leaky_relu(inp + self.elinear(inp))
            return out
        else:
            spk, mem = self.sleaky(inp + self.elinear(inp), mem)
            return spk, mem



class AudioCortex(Basic):
    def __init__(
            self,
            in_features,
            out_features,
    ):
        super(AudioCortex, self).__init__(in_features, out_features)
        self.sleaky = snn.Leaky(beta=0.95)
        self.flinear = nn.Linear(in_features, out_features, bias=False)
        # self.leaky_relu = nn.LeakyReLU()
    def forward(self, inp, mem):
        spk, mem = self.sleaky(self.flinear(inp), mem)
        return spk, mem
