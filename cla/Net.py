import copy

import torch.nn as nn
from cla.Layer import AuditoryNerve,Bushy,InferiorColliculus,AudioCortex,Cochlear,InnerHairCell
import snntorch as snn
import torch
import numpy as np
from spikingjelly.activation_based import neuron

# update_rule

# if_layer = neuron.IFNode()
# if_layer.reset()


import matplotlib.pyplot as plt
from cla.Data_process import Data_process as dp
# torch.uniform(weight, -stdv, stdv)

class Net(nn.Module):
    def __init__(self, in_features, hidden, out_features, model, **kwargs):
        super().__init__()
        self.model = model
        self.ic_mode = kwargs['ic_mode']
        self.cochlear = Cochlear(channels=in_features, sample_rate=kwargs['sample_rate'],frequency_range=kwargs['frequency_range'],
                                 bandwidth_factor=kwargs['bandwidth_factor'])
        self.ihc = InnerHairCell(window=kwargs['window'], stride=kwargs['stride'])
        self.spk_bushy_stack=[]
        self.spk_ic_stack=[]
        #
        if model in ['lstm','rnn','gru']:
            self.ic = InferiorColliculus(50, 50)
            self.fc = nn.Linear(50, 10, bias=False)
            self.lif = snn.Leaky(beta=0.95)
            self.ac = AudioCortex(50, 10)
            if model=='lstm':
                self.lstm = nn.LSTM(in_features,50,batch_first=True)
            elif model =='rnn':
                self.rnn = nn.RNN(in_features,50,batch_first=True)
            elif model =='gru':
                self.gru = nn.GRU(in_features,50,batch_first=True)

        elif model == 'mh':
            self.an = AuditoryNerve(ans=kwargs['ans'],range=kwargs['an_range'])
            self.bushy =Bushy(in_features*kwargs['ans'],hidden,scale_range=kwargs['bushy_range'])
            if kwargs['ic_mode']:
                self.ic = InferiorColliculus(hidden,hidden)
            self.ac = AudioCortex(hidden,out_features)

    def forward(self, inp):
        spk_an_rec = []
        spk_bushy_rec = []
        spk_ic_rec = []
        spk_ac_rec = []
        mem_ac_rec = []

        # plt.plot(inp[0,:,20].detach().numpy())

        inp = self.cochlear(inp)
        inp = self.ihc(inp)

        if self.model in ['lstm','gru','rnn']:
            mem_out = self.lif.init_leaky()
            if self.model== 'lstm':
                inp, (h_n, c_n) = self.lstm(inp)
            elif self.model =='rnn':
                inp, hn = self.rnn(inp)
            elif self.model =='gru':
                inp, h_n = self.gru(inp)

            for t in range(inp.size(1)):
                out = inp[:,t,:]

                # out = self.ic(out)

                spk_out = self.fc(out)
                mem_out= spk_out

                # spk_out,mem_out = self.ac(out,mem_out)

                # spk_out,mem_out = self.lif(out,mem_out)
                spk_ac_rec.append(spk_out)
                mem_ac_rec.append(mem_out)
            self.spk_bushy_stack = torch.rand(3,3,3)
            self.spk_ic_stack = torch.rand(3,3,3)

            out = (torch.stack(spk_ac_rec, dim=1), torch.stack(mem_ac_rec, dim=1))
            # out = (inp, inp)

        elif self.model in['mh']:
            if self.ic_mode:
                mem_bushy = self.bushy.sleaky.init_leaky()
                mem_ic = self.ic.sleaky.init_leaky()
                mem_ac = self.ac.sleaky.init_leaky()
                for t in range(inp.size(1)):
                    spk_an = self.an(inp[:, t, :])
                    spk_bushy, mem_bushy = self.bushy(spk_an, mem_bushy)

                    spk_ic, mem_ic = self.ic(spk_bushy, mem_ic)
                    spk_ac, mem_ac = self.ac(spk_ic, mem_ac)

                    spk_an_rec.append(spk_an)
                    spk_bushy_rec.append(spk_bushy)
                    spk_ic_rec.append(spk_ic)
                    spk_ac_rec.append(spk_ac)
                    mem_ac_rec.append(mem_ac)
                self.spk_bushy_stack = torch.stack(spk_bushy_rec, dim=1)
                self.spk_ic_stack = torch.stack(spk_ic_rec, dim=1)
                out = (torch.stack(spk_ac_rec, dim=1), torch.stack(mem_ac_rec, dim=1))
            else:
                mem_bushy = self.bushy.sleaky.init_leaky()
                mem_ac = self.ac.sleaky.init_leaky()

                for t in range(inp.size(1)):
                    spk_an = self.an(inp[:, t, :])
                    spk_bushy, mem_bushy = self.bushy(spk_an, mem_bushy)
                    spk_ac, mem_ac = self.ac(spk_bushy, mem_ac)

                    spk_an_rec.append(spk_an)
                    spk_bushy_rec.append(spk_bushy)
                    spk_ac_rec.append(spk_ac)
                    mem_ac_rec.append(mem_ac)
                self.spk_bushy_stack = torch.stack(spk_bushy_rec, dim=1)
                self.spk_ic_stack = torch.rand(3, 3, 3)
                out = (torch.stack(spk_ac_rec, dim=1), torch.stack(mem_ac_rec, dim=1))
            # plt.imshow(self.spk_ic_stack.detach().numpy(),aspect='auto')
        return out

        # spk_an_stack = torch.stack(spk_an_rec, dim=1)
        # a = spk_an_stack[0, :, :].detach().numpy()
        # plt.figure()
        # plt.imshow(a, aspect='auto')
        # plt.title('spk_sgc')
            # inp = torch.stack(spk_bushy_rec, dim=1)

        # for t in range(inp.size(1)):
        #     spk_ic, mem_ic = self.ic(inp[:, t, :], mem_ic)
        #     spk_ac, mem_ac = self.ac(spk_ic, mem_ac)
        #
        #     spk_ic_rec.append(spk_ic)
        #     spk_ac_rec.append(spk_ac)
        #     mem_ac_rec.append(mem_ac)
        # out = (torch.stack(spk_ac_rec, dim=1), torch.stack(mem_ac_rec, dim=1))





        # elif self.mode =='fc':
        #     mem_lif0 = self.lif0.init_leaky()
        #     spk_bushy_rec = []
        #     spk_ic_rec =[]
        #     spk_ac_rec = []
        #     mem_ac_rec = []
        #
        #     for t in range(inp.size(1)):
        #         out_fc0 = self.fc0(inp[:,t,:])
        #         spk_bushy, mem_lif0 = self.lif0(out_fc0,mem_lif0)
        #         spk_ic, mem_ic = self.ic(spk_bushy, mem_ic)
        #         spk_ac, mem_ac = self.ac(spk_ic, mem_ac)
        #
        #         spk_bushy_rec.append(spk_bushy)
        #         spk_ic_rec.append(spk_ic)
        #
        #         spk_ac_rec.append(spk_ac)
        #         mem_ac_rec.append(mem_ac)
        #     out = (torch.stack(spk_ac_rec, dim=1), torch.stack(mem_ac_rec, dim=1))
        #     self.spk_bushy_stack = torch.stack(spk_bushy_rec, dim=1)
        #     self.spk_ic_stack = torch.stack(spk_ic_rec, dim=1)

        # elif self.model == 'mh':
        #     mem_bushy = self.bushy.sleaky.init_leaky()
        #
        #     spk_bushy_rec = []
        #     spk_ic_rec = []
        #     spk_ac_rec = []
        #     mem_ac_rec = []
        #
        #     # cochlear_out = self.cochlear(inp)
        #     # ihc_out = self.ihc(inp,mode='half')
        #     for t in range(inp.size(1)):
        #         # spk_sgc = self.sgc(inp[:,t,:])
        #         spk_bushy,mem_bushy= self.bushy(inp[:,t,:], mem_bushy)
        #         spk_ic, mem_ic = self.ic(spk_bushy, mem_ic)
        #         spk_ac, mem_ac = self.ac(spk_ic, mem_ac)
        #         # plt.imshow(spk_sgc[0],aspect='auto')
        #         # spk_sgc_rec.append(spk_sgc)
        #         spk_bushy_rec.append(spk_bushy)
        #         spk_ic_rec.append(spk_ic)
        #
        #         spk_ac_rec.append(spk_ac)
        #         mem_ac_rec.append(mem_ac)
        #     out = (torch.stack(spk_ac_rec, dim=1), torch.stack(mem_ac_rec, dim=1))
        #     self.spk_bushy_stack = torch.stack(spk_bushy_rec, dim=1)
        #     self.spk_ic_stack = torch.stack(spk_ic_rec, dim=1)
            # plt.imshow(out[0][0,:,:].detach().numpy(),aspect='auto')

        # import copy
        # a = copy.deepcopy(self.fc.weight[0,:].detach().numpy())
        # plt.figure()
        # # ind = (np.where(a<0))
        # # a[ind]=0
        # plt.plot(a)

        # plt.imshow()
        # spk_sgc_stack = torch.stack(spk_sgc_rec, dim=1)
        # spk_bushy_stack = torch.stack(spk_bushy_rec, dim=1)
        # spk_ic_stack = torch.stack(spk_ic_rec, dim=1)
        # # spk_ac_stack = torch.stack(spk_ac_rec, dim=1)
        # # mem_ac_stack = torch.stack(mem_ac_rec, dim=1)
        # pic = 3
        # plt.imshow(spk_bushy_stack.detach().numpy()[pic,:,:],aspect='auto')
        # plt.figure()
        # plt.imshow(spk_ic_stack.detach().numpy()[pic, :, :], aspect='auto')
        # mean_after=[]
        # mean_before = []
        # # data = spk_bushy_stack # spk_bushy_stack  spk_ic_stack
        # for i in range(40):
        #     temp = spk_bushy_stack[i,:,:].detach().numpy()
        #     std_x = np.std(temp,axis=1)
        #     std_x = std_x.reshape((std_x.shape[0], 1))
        #     std_mat = np.matmul(std_x,std_x.transpose())
        #     corcoe = np.cov(temp)/(std_mat+1e-10)
        #     mean_before.append(np.mean(corcoe))
        #
        #
        #     temp = spk_ic_stack[i,:,:].detach().numpy()
        #     std_x = np.std(temp,axis=1)
        #     std_x = std_x.reshape((std_x.shape[0], 1))
        #     std_mat = np.matmul(std_x,std_x.transpose())
        #     corcoe = np.cov(temp)/(std_mat+1e-10)
        #     mean_after.append(np.mean(corcoe))
        #     # print(mean_corcoe)
        # mean_before = np.mean(mean_before)
        # mean_after = np.mean(mean_after)
        # print(mean_before)
        # print(mean_after)

        #
        # plt.imshow(corcoe,aspect='auto')
        # plt.imshow(std_mat, aspect='auto')
        # std_y = np.std(temp,axis=1)


        # for i in range(temp.shape[0]):

        #
        # pic=4
        # index = torch.where(pic==label)
        # for ind in index[0]:
        #     a = spk_sgc_stack[ind, :, :].detach().numpy()
        #     plt.figure()
        #     plt.imshow(a,aspect='auto')
        #     plt.title('spk_sgc')


            # #
            # pic=2
            # # a,_= torch.max(inp[pic,:,:],dim=0)
            # # plt.figure()
            # # plt.plot(a)
            # #

            # # #
            # a = spk_bushy_stack[pic, :, :].detach().numpy()
            # plt.figure()
            # plt.imshow(a,aspect='auto')
            # plt.title('spk_bushy')
            #
            # a = spk_ic_stack[pic, :, :].detach().numpy()
            # plt.figure()
            # plt.imshow(a,aspect='auto')
            # plt.title('spk_ic')
            #
            # a = spk_ac_stack[pic, :, :].detach().numpy()
            # plt.figure()
            # plt.imshow(a,aspect='auto')
            # plt.title('spk_ac')
            #
            # a = mem_ac_stack[pic, :, :].detach().numpy()
            # plt.figure()
            # plt.imshow(a,aspect='auto')
            # plt.title('mem_ac')



            #
            # a = spk1_stack[pic, :, :].detach().numpy()
            # plt.figure()
            # plt.imshow(a,aspect='auto')
            # plt.title('spk1')
            #
            # a = out[0][pic, :, :].detach().numpy()
            # plt.figure()
            # plt.imshow(a,aspect='auto')
            # plt.title('output 0')



            # a = cur0_stack[0,:,:].detach().numpy() # inp spk1_stack cur0_stack
            # b = np.corrcoef(a.transpose())
            # ind = np.where(np.isnan(b))
            # b[ind]=0
            # plt.imshow(b)
            # plt.title('input_corcoe 0.5-2 500')

            # a = inp[0,:,:].detach().numpy() # spk1_stack cur0_stack
            # plt.figure()
            # plt.imshow(a,aspect='auto')
            # b = np.sum(a, axis=0)
            # # plt.figure()
            # # plt.plot(b)
            #
            # b_max = np.max(b)
            # b_norm = b/b_max
            # plt.figure()
            # plt.plot(b_norm)
            #
            # b_std = np.std(b_norm)
            # print(b_std)
            #
            # sum=[]
            # for i in range(0,b_norm.shape[0],int(0.1*b_norm.shape[0])):
            #     sum.append(np.sum(b_norm[i:i+int(0.1*b_norm.shape[0])]))
            # m = np.array(sum)
            # plt.figure()
            # plt.plot(m/m.max())
            # print(np.std(m))

            # plt.figure()
            # w = self.rs1.plinear1.w[0, :].detach().numpy()
            # plt.plot(w)
            # ave_w=[]
            # for i in range(0,w.shape[0],int(0.01*w.shape[0])):
            #     ave_w.append(np.sum(w[i:i+int(0.01*w.shape[0])]))
            # plt.figure()
            # plt.plot(np.array(ave_w))
            # m = np.array(ave_w)

            # b_max = np.max(b, axis=1, keepdims=True)
            # plt.figure()
            # plt.plot((b / b_max)[0, :])
            # b_std = np.std(b / b_max)
            # plt.plot(b[0, :])

            # a = spk1_stack.detach().numpy()
            # b = np.mean(a, axis=1)
            # b_max = np.max(b, axis=1, keepdims=True)
            # plt.figure()
            # plt.plot((b / b_max)[0, :])
            # b_std = np.std(b / b_max)
            # plt.figure()
            # plt.plot(b[0, :])

            # plt.imshow(out[0][6, :, :].detach().numpy(), aspect='auto')



