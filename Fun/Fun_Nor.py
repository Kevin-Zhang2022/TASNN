import numpy as Num
import numpy.random


def Fun_Nor(Dat_Org, Mod):
    if Mod == 'Max_Min':
        Sha = Num.shape(Dat_Org)
        if len(Sha) == 1:
            Dat_Org = Num.reshape(Dat_Org, [1, Sha[0]])
        Min = Num.min(Dat_Org, 1)
        Max = Num.max(Dat_Org, 1)
        Dat_Nor = (Dat_Org - Min)/(Max-Min)
        return Dat_Nor

    elif Mod == 'Abs':
        Sha = Num.shape(Dat_Org)
        if len(Sha) == 1:
            Dat_Org = Num.reshape(Dat_Org, [1, Sha[0]])
        Dat_Nor = (Dat_Org - (-3))/(3-(-3))
        return Dat_Nor




