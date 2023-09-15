

import numpy as Num
from Lay.ClaPar import ClaPar as Par
import cv2

def Fun_Rec(Inp):
    Sca1 = 0.625
    Sca2 = 0.125
    Sca3 = -0.125
    Sca4 = -0.5

    Wei = [[Sca4, Sca3, Sca2, Sca3, Sca4],
           [Sca3, Sca2, Sca1, Sca2, Sca3],
           [Sca2, Sca1, 1, Sca1, Sca2],
           [Sca3, Sca2, Sca1, Sca2, Sca3],
           [Sca4, Sca3, Sca2, Sca3, Sca4]]

    Pot = Num.zeros([Par.PicSiz[0], Par.PicSiz[1]])
    Ran = [-2, -1, 0, 1, 2]  # range
    X_Cor = 2
    Y_Cor = 2

    for i in range(Par.PicSiz[0]):
        for j in range(Par.PicSiz[1]):
            Sum = 0
            for m in Ran:
                for n in Ran:
                    if 0 <= (i+m) <= Par.PicSiz[0]-1 and 0 <= (j+n) <= Par.PicSiz[1]-1:
                        Sum = Sum + Wei[X_Cor+m][Y_Cor+n]*Inp[i+m][j+n]/255
            Pot[i][j] = Sum
    return Pot


if __name__ == '__main__':

    Img = cv2.imread('../Dat/' + str(1) + '.png', 0)
    Pot = Fun_Rec(Img)

    Max_Arr = []
    Min_Arr = []

    for i in Pot:
        Max_Arr.append(max(i))
        Min_Arr.append(min(i))
    print('Max', max(Max_Arr))
    print('Min', min(Min_Arr))
















