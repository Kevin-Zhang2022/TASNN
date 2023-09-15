import numpy as Num
import numpy as np
from scipy.fftpack import fft
# from Fun.Fun_Shu import Fun_Shu
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


class normal(nn.Module):
    # out = torch.zeros(inp.size(0), inp.size(1)*sgcs, inp.size(2))
    # for b in range(inp.size(0)):
    def __init__(
        self,
        mode='01',
    ):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        # a = F.normalize(x,dim=1)

        if self.mode =='-11':
            mean = torch.mean(x, dim=0)
            std = torch.std(x, dim=0)
            out = (x-mean)/(std+1e-12)

            # valve_array = torch.arange(0.05, 0.05+1, 1/self.sgcs).unsqueeze(0).unsqueeze(0).tile((1,1,x.size(2)))
            # inp = torch.repeat_interleave(x0, dim=2, repeats=self.sgcs)
            # out = (inp>valve_array).float()
        return out

    # plt.plot(out.detach().numpy()[0,:])
    # plt.imshow(out[0,:,:])
    # a = valve_array.unsqueeze(0).unsqueeze(2).tile((inp.size(0), inp.size(1), inp.size(2)))
    # inp = inp.tile(1,sgcs,1)
    # a =10
    # a = out[0,250:260,:].numpy().sum(0)
    # plt.plot(a)
    # inp = torch.tensor(inp,dtype=torch.float32)
    # index = torch.where(inp<2e-5)
    # inp[index] = 2e-5
    # inp = 20*torch.log10((inp/2e-5))
    # inp = inp/100
    # total_batch = int((inp.size(1)-window)/stride+1)
    # outx = torch.zeros(total_batch, inp.size(0), window)
    # for j in range(total_batch):
    #     outx[j, :, :] = inp[:, j*stride:j*stride+window]
    #     # out.append(label)
    # # plt.plot(inp[25,:])
    # outy = torch.ones(total_batch)*label








