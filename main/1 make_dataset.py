from cla.Data_set import Data_set as ds
from torch.utils.data import DataLoader
from Fun.GTFB import GTFB
import torch
import matplotlib.pyplot as plt
from Fun.norm import norm


if __name__ == '__main__':
    # get_data_list('dataset/audio', 'dataset')
    sample_rate = 20000
    band_width = [0, 10000]
    channels = 200
    batch_size = 20

    train_dataset = ds(data_list_path='../data/car_diagnostics/train_list.csv',
                       max_duration=1, sample_rate=sample_rate, use_dB_normalization=False)
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)

    torch.save(train_loader, 'train_loader.pth')
    # train_loader1 = torch.load('train_loader.pth')
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # data_type = torch.float32
    # loss = nn.CrossEntropyLoss()
    #
    # net = snn_net(in_feature=1000, my=True)
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.1, betas=(0.9, 0.999))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    fig0 = plt.figure(figsize=(10, 6))
    axe = []
    for i in range(10):
        axe.append(fig0.add_subplot(5, 2, i + 1))

    for batch_id, (audio, label) in enumerate(train_loader):
        # plt.plot(audio[0,:])
        # plt.plot(data[2,:])
        data = norm(audio, mode='-11')
        data, cf = GTFB(data, band_width=band_width, channels=channels, sam_rate=sample_rate)
        a, _ = torch.max(data, dim=1)

        for i in range(batch_size):
            axe[int(label[i])].plot(cf,a[i,:])



    a=10
