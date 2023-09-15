import numpy as Num


def Fun_DatSeg(SpiMap_All, Win, Rat_Tra, Lab):
    Bat = Num.shape(SpiMap_All)[1] - Win
    SpiMap = Num.zeros([Bat, Num.shape(SpiMap_All)[0], Win, 1])

    for j in range(0, Num.shape(SpiMap_All)[1]-Win):
        SpiMap[j, :, :, 0] = SpiMap_All[:, j:j + Win]

    SpiMap_TraX = SpiMap[0:int(Rat_Tra*Bat), :, :,: ]
    SpiMap_TraY = Lab * Num.ones(Num.shape(SpiMap_TraX)[0])
    SpiMap_TesX = SpiMap[int(Rat_Tra * Bat): Bat, :, :, :]
    SpiMap_TesY = Lab * Num.ones(Num.shape(SpiMap_TesX)[0])

    return SpiMap_TraX, SpiMap_TraY, SpiMap_TesX, SpiMap_TesY




