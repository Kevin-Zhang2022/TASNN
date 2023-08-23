import numpy as np
from cla.Data_set import Data_set as ds
from torch.utils.data import DataLoader
from cla.Net import Net
import torch
import torch.nn as nn
from cla.Show import Show as show
from Fun.norm import norm
from cla.Data_process import Data_process as dp
import matplotlib.pyplot as plt
from openpyxl import Workbook
import random
from scipy import stats
import os


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(20)

batch_size = 40

task = 'car_diagnostics'

mode='mh' # 'lstm' 'gru' 'rnn' 'fc' 'mh'

par_list =[2]

# par_list = [[0.5,4.15],[0.5,5.35],[0.5,6.55],[0.5,7.70]]
# par_list = [[0.5,9.7],[0.5,11.7],[0.5,13.5],[0.5,15.1]]
# train

spk_ic_stack=[]
spk_bushy_stack=[]
labels=[]
for par in par_list:
    dp.create_datalist(path2audio='../data/'+task+'/audio')
    train_dataset = ds(data_list_path='../data/'+task+'/train_list.csv',
                       max_duration=1, sample_rate=10000, use_dB_normalization=False)
    test_dataset = ds(data_list_path='../data/'+task+'/test_list.csv',
                      max_duration=1, sample_rate=10000, use_dB_normalization=False)
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=20)
    net = torch.load('../net_for_figtab/average cc net.pkl')

    for batch_id, (audio, label) in enumerate(train_loader):
        data = dp.normalize(audio)
        out = net(data)
        labels=label
    spk_ic_stack = net.spk_ic_stack.detach().numpy()
    spk_bushy_stack = net.spk_bushy_stack.detach().numpy()
    weights=net.ic.elinear.w.detach().numpy()

# plt.plot(weights)
fig = plt.figure(figsize=(8,4))
fontsize=8
# title_list=['a','b','c','d','e','f','g','h']

i=20
spk_bushy = spk_bushy_stack[i,:,:]
spk_ic = spk_ic_stack[i,:,:]

axe = fig.add_subplot(1,2,1)
axe.imshow(spk_bushy, aspect='auto')
# axe.set_title('(a) Average correlation coefficient=0.00386', color='black',y=-0.25,fontsize=fontsize)
# axe.set_xlabel('Input channel ordinal', fontsize=fontsize)
# axe.set_ylabel('Output channel ordinal', fontsize=fontsize)

axe = fig.add_subplot(1,2,2)
axe.imshow(spk_ic, aspect='auto')
# axe.set_title('(b) Average correlation coefficient=0.00197', color='black',y=-0.25,fontsize=fontsize)
# axe.set_xlabel('Input channel ordinal', fontsize=fontsize)
# axe.set_ylabel('Output channel ordinal', fontsize=fontsize)

fig.subplots_adjust(left=0.05,bottom=0.1,right=0.98,top=0.975,wspace=0.224,hspace=0.338)
fig.show()


spk_bushy = np.tile(np.expand_dims(spk_bushy.transpose(), axis=0),[40,1,1])
bushy_cc = dp.average_ccmat(torch.tensor(spk_bushy))
spk_ic = np.tile(np.expand_dims(spk_ic.transpose(), axis=0),[40,1,1])
ic_cc = dp.average_ccmat(torch.tensor(spk_ic))

filename = '../fig/E3 IC/E3 Average CC before and after IC circuit.tif'
fig.savefig(filename)




