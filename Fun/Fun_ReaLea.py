from Cla.Par import Par
import numpy as Num

# STDP reinforcement learning curve
def Fun_ReaLea(t):
    if t > 0:
        return -Par.A_Pos * Num.exp(-float(t) / Par.Tao_Pos)
    if t <= 0:
        return Par.A_Neg * Num.exp(float(t) / Par.Tao_Neg)