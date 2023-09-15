import numpy as np

from cla.Data_set import my_dataset
from torch.utils.data import DataLoader
from cla.Net import snn_net
from Fun.GTFB import GTFB
import torch
import torch.nn as nn
from Fun.norm import norm
from Fun.PeakValue import PeakValue
from Fun.seg_data import create_datalist

if __name__ == '__main__':
    sample_rate = 10000
    band_width = [0, 5000]
    channels = 200
    batch_size = 20

    large_cycle = 1
    reps = 1
    epochs = 20
    sc_step_size = 5

    # train_loader = torch.load('train_loader.pth')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_type = torch.float32
    loss = nn.CrossEntropyLoss()

    # train
    train_AA_i = []
    train_AL_i = []
    test_AA_i = []
    test_AL_i = []
    for i in range(large_cycle):
        create_datalist(path2audio='../data/car_diagnostics/audio')
        train_dataset = my_dataset(data_list_path='../data/car_diagnostics/train_list.csv',
                                   max_duration=1, sample_rate=sample_rate, use_dB_normalization=False)
        test_dataset = my_dataset(data_list_path='../data/car_diagnostics/test_list.csv',
                                  max_duration=1, sample_rate=sample_rate, use_dB_normalization=False)
        train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size)
        net = snn_net(in_feature=channels)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.1, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sc_step_size, gamma=0.1)
        for epoch in range(epochs):
            for batch_id, (audio, label) in enumerate(train_loader):
                data = norm(audio, mode='-11')
                data, cf = GTFB(data, band_width=band_width, channels=channels, sam_rate=sample_rate)

                data = PeakValue(data)
                out = net(data)
                loss_val = loss(out, label)
                # plt.plot(data[0,:,25])
                # plt.plot(a)
                # plt.imshow(data[1,0:1000,:])
                # for i in range(8):
                #     a,_ = data[i].max(0)
                #     plt.plot(a)

                # Gradient calculation + weight update
                optimizer.zero_grad()
                loss_val.backward(retain_graph=True)
                optimizer.step()

                _, predicted = out.max(1)
                total = label.size(0)
                correct = (predicted == label).sum().item()
                acc = correct/total

                print(f'training i:{i:}', f"epoch: {epoch:d}", f"acc: {acc:.2f}", f"loss: {loss_val:.2f}")
                # print(f"lr: {optimizer.state_dict()['param_groups'][0]['lr']:.2f}")
                # print('epoch:', epoch,'acc:%.2f',correct/total,'loss:',loss_val)
            scheduler.step()

        # test train
        train_loss = []
        train_acc = []
        for rep in range(reps):
            for batch_id, (audio, label) in enumerate(train_loader):
                data = norm(audio, mode='-11')
                data, cf = GTFB(data, band_width=band_width, channels=channels, sam_rate=sample_rate)

                data = PeakValue(data)
                out = net(data)
                loss_val = loss(out, label)

                _, predicted = out.max(1)
                total = label.size(0)
                correct = (predicted == label).sum().item()
                acc = correct / total
                train_acc.append(acc)
                train_loss.append(loss_val.detach().numpy())
                print(f"test train i: {i:d}", f"rep: {rep:d}", f"acc: {acc:.2f}", f"loss: {loss_val:.2f}")
        train_AA_i.append(np.mean(np.array(train_acc)))
        train_AL_i.append(np.mean(np.array(train_loss)))

        # test test
        test_loss = []
        test_acc = []
        for rep in range(reps):
            for batch_id, (audio, label) in enumerate(test_loader):
                data = norm(audio, mode='-11')
                data, cf = GTFB(data, band_width=band_width, channels=channels, sam_rate=sample_rate)

                data = PeakValue(data)
                out = net(data)
                loss_val = loss(out, label)
                _, predicted = out.max(1)
                total = label.size(0)
                correct = (predicted == label).sum().item()
                acc = correct/total

                test_acc.append(acc)
                test_loss.append(loss_val.detach().numpy())

            print(f'test_test', f'i:{i:d}', f'rep{rep:d}', f"acc: {acc:.2f}", f"loss: {loss_val:.2f}")
        test_AA_i.append(np.mean(np.array(test_acc)))
        test_AL_i.append(np.mean(np.array(test_loss)))

    # np.mean(train_AA_i)
    # np.mean(train_AL_i)
    # np.mean(test_AA_i)
    # np.mean(test_AL_i)
    a=10

    # acc_all = np.array(acc_all)
    # test_acc = np.mean(acc_all)
    # loss_all = np.array(loss_all)
    # test_loss = np.mean(loss_all)
    # print(f'i:{i:d}',f'test_acc:{test_acc:.2f}',f'test_loss:{test_loss:.2f}')
