import torch
import matplotlib.pyplot as plt
import librosa

def ave_magnitude(inp):
    out = []
    for i in range(0,inp.size(1),20):
        a = torch.mean(abs(inp[:, i:i + 200,:]),dim=1)
        out.append(a.unsqueeze(1))
    out = torch.cat(out, dim=1)


    #
    # plt.figure()
    # plt.plot(inp[0,:,25])
    # plt.figure()
    # plt.plot(out_low[0,:,25])
    #
    # dif1_0 = inp[:,1:-1,:]-inp[:,0:-2,:]
    # dif1_2 = inp[:,1:-1,:]-inp[:,2:,:]
    # temp = torch.zeros(inp.size(0),1,inp.size(2))
    #
    # dif = (dif1_0>0)*(dif1_2>0)
    # dif = torch.cat([temp, dif],dim=1)
    # dif = torch.cat([dif,temp], dim=1)
    #
    # # a = torch.sum(dif,dim=1)
    # # b,_= torch.max(a,dim=0)
    # # plt.plot(b)
    # out = dif*inp
    #
    # c = torch.where(out>0)
    # plt.imshow(dif[0,:])
    # plt.imshow(out[0,:])
    # plt.plot(out[0,:,25])
    return out

