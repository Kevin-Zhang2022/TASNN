import matplotlib.pyplot as plt
import torch.nn as nn
from cla.Layer import AuditoryNerve,Bushy,InferiorColliculus,AudioCortex,Cochlear,InnerHairCell
import snntorch as snn
import torch
import numpy as np
from spikingjelly.activation_based import neuron

in_features=200
frequency_range=[5,5000]
bandwidth_factor=0.05
sample_rate=10000


cochlear = Cochlear(channels=in_features, frequency_range=frequency_range,
                    bandwidth_factor=bandwidth_factor,sample_rate=sample_rate)

cf = cochlear.cf[10]

x0=torch.arange(0,1,1/sample_rate)

x1 = torch.sin(2*np.pi*cf*x0)
# plt.plot(x1)
# plt.show()

x2=x1.unsqueeze(0).tile([2,1])

y=cochlear(x2)

y_10 = y[0,:,10].detach().numpy()
y_9 = y[0,:,9].detach().numpy()
y_11 = y[0,:,9].detach().numpy()

plt.figure()
plt.plot(y_9)
plt.figure()
plt.plot(y_10)
plt.figure()
plt.plot(y_11)


a=10