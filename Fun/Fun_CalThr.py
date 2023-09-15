

import numpy as Num
from Lay.ClaPar import ClaPar as Par

def Fun_CalThr(Inp):
    SpiTra = Inp
    Row = Num.shape(SpiTra)[1]
    Thr = 0

    for i in range(Row):
        Sum = sum(SpiTra[:, i])
        if Sum > Thr:
            Thr = Sum

    return (Thr/3)*Par.Sca


# if __name__ == '__main__':

    # Img = cv2.imread('../Dat/' + str(1) + '.png', 0)
    # Pot = Fun_Rec(Img)
    #
    # Max_Arr = []
    # Min_Arr = []
    #
    # for i in Pot:
    #     Max_Arr.append(max(i))
    #     Min_Arr.append(min(i))
    # print('Max', max(Max_Arr))
    # print('Min', min(Min_Arr))
















