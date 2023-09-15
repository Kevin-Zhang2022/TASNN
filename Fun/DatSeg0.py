import numpy as Num
import numpy as np
from scipy.fftpack import fft
from Fun.Fun_Shu import Fun_Shu
import matplotlib.pyplot as plt
import torch
# win > fs * 1/ft(target)  1000*(1/50~1/250) -> 4~20 point*5 -> 80~100
# stride = 1
def DatSeg0(inp, window, stride, label):
    # out = []
    # out = []
    inp = torch.tensor(inp,dtype=torch.float32).t()
    # torch.max(inp)
    inp = inp/3
    total_batch = int((inp.shape[0]-window)/stride+1)
    outx = torch.zeros(total_batch,window,inp.size(1))
    for j in range(total_batch):
        outx[j, :, :] = inp[j*stride:j*stride+window, :]
        # out.append(label)
    # out0 = np.array(out,dtype=float)
    # out0= torch.tensor(,dtype=torch.float32)
    # outx = out0.reshape[out0.shape[0],1,1,out0.shape[1]]
    outy = torch.ones(total_batch)*label
    # a = outx.reshape(outx.shape[3],outx.shape[0],outx.shape[1],outx.shape[2],1)
    # outx.shape[3]
    # plt.plot(a[:,0,0,23,0])



    return outx, outy


    # if Mod == 'Fre':
    #     Siz_Inp = np.shape(Inp)
    #     Col_Gen = Siz_Inp[1]
    #     Row_Map = Siz_Inp[0]
    #     Col_Map = Win
    #     Bat_Map = Col_Gen - Win
    #
    #     # Siz_Map = [Bat_Map, Row_Map, Col_Map, 1]
    #     Map_Out_0 = Num.zeros([Bat_Map, Siz_Inp[0], Win, 1])  # put segmented map
    #     FreMap_Out_0 = Num.zeros([Bat_Map, Siz_Inp[0], Col_Map, 1])  # put original frequency map
    #     FreMap_Out_1 = Num.zeros([Bat_Map, Siz_Inp[0], Fre_Foc[1] - Fre_Foc[0], 1])  # put focused frequency map
    #     if Mod == 'Fre':
    #         for j in range(0, Bat_Map):
    #             Map_Out_0[j, :, :, 0] = Inp[:, j:j + Win]
    #             FreMap_Out_0[j, :, :, 0] = abs(fft(Map_Out_0[j, :, :, 0]))/(Win/2)
    #             FreMap_Out_1[j, :, :, 0] = FreMap_Out_0[j, :, Fre_Foc[0]:Fre_Foc[1], 0]
    #             # import matplotlib.pyplot as plt
    #             # plt.plot(FreMap_Out_0[0, 0, :, 0])
    #             # plt.plot(FreMap_Out_1[0, 0, :, 0])
    #
    #     FreMap_Out_1 = Fun_Shu(FreMap_Out_1)
    #     SpiMap_TraX = FreMap_Out_1[0:int(Rat_Tra * Bat_Map), :, :, :]
    #     SpiMap_TraY = Lab * Num.ones(Num.shape(SpiMap_TraX)[0])
    #     SpiMap_TesX = FreMap_Out_1[int(Rat_Tra * Bat_Map): Bat_Map, :, :, :]
    #     SpiMap_TesY = Lab * Num.ones(Num.shape(SpiMap_TesX)[0])






