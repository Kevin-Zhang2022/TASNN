
import numpy as Num
import matplotlib.pyplot as Plt
def Fun_Shu(SpiMap_Inp):
    Pag = Num.shape(SpiMap_Inp)[0]
    Seq = Num.arange(0, Pag)
    # Num.random.seed(1024)
    Num.random.shuffle(Seq)
    SpiMap_Out = SpiMap_Inp[Seq, :, :]
    return SpiMap_Out

# Plt.plot(SpiMap_Out[0].squeeze())