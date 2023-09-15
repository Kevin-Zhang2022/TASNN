import random

import numpy as Num
import struct

import math


def Fun_Enc(Dat_Nor, Mod_Enc, Len_Enc):
    if Mod_Enc == 'Rat':
        Sha = Num.shape(Dat_Nor)
        Dat_Enc = Num.zeros([Sha[0], Sha[1] * Len_Enc])
        # Num.random.seed(1024)
        RanMat = Num.random.random([Sha[0], Sha[1] * Len_Enc])
        Ext_DatNor = Dat_Nor.repeat(Len_Enc, axis=1)
        Ind_Fir = Num.where(Ext_DatNor > RanMat)
        Dat_Enc[Ind_Fir] = 1

    return Dat_Enc


# array = Numpy.zeros((3,10))
# b = [[1,2,3,],[4,5,6],[7,8,9]]
# array[0:3,0:3] = b


# if __name__ == '__main__':
#
#     Img = cv2.imread("../Dat/" + str(1) + ".png", 0)
#     Pot = Fun_Rec(Img)
#
#     # print max(m), min(n)
#     SpiTra = Fun_Enc(Pot)
#     f = open('../Dat/train1.txt', 'w')
#     print(Num.shape(SpiTra))
#
#     for i in range(201):
#         for j in range(784):
#             f.write(str(int(SpiTra[j][i])))
#         f.write('\n')
#     f.close()
