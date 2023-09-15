import numpy as np
import torch
import matplotlib.pyplot as plt

def norm(inp,mode='-11'):
    if mode == '-11':
        out = (inp-inp.mean(1).unsqueeze(1))/torch.std(inp,dim=1).unsqueeze(1)
    return out

    # plt.plot(out[3,:])
