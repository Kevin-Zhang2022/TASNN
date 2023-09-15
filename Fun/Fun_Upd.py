# This file implements STDP curve and weight update rule


import numpy as Num
from matplotlib import pyplot as plt
from Cla.Par import Par


def Fun_Upd(W, Del_W):
    if Del_W < 0:
        return W + Par.LeaRat * Del_W * (W - abs(Par.W_Min)) * Par.Sca
    elif Del_W > 0:
        return W + Par.LeaRat * Del_W * (Par.W_Max - W) * Par.Sca


# if __name__ == '__main__':
#     print(Fun_Get_DelWei(-20) * Par.Sig)

