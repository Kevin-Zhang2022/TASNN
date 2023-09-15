import numpy as np
# from Cell.LIF import LIF
import pyfilterbank.gammatone as gt

def GTF(inp,band_width,channels,sam_rate):
    start_band = gt.hertz_to_erbscale(band_width[0])
    end_band = gt.hertz_to_erbscale(band_width[1])

    GTF_bank = gt.GammatoneFilterbank(samplerate=sam_rate, bandwidth_factor=0.05, order=4, startband=start_band,
                                      endband=end_band,density=(end_band-start_band)/channels, normfreq=0)

    # cf = GTF_bank.centerfrequencies
    results = GTF_bank.analyze(inp)
    temp = []
    # IHC_R_real = []
    for (band, state) in results:
        temp.append(np.real(band))
    temp = np.array(temp)
    # temp[np.where(temp<=2e-5)] = 2e-5

    # out = np.array(out)
    # out1 = torch.tensor(out, dtype=torch.float32).reshape(out.shape[0], 1, out.shape[1], out.shape[2])
    # mask = out1>0
    # out2 = out1*mask

    # m = temp
    # a = np.max(m, axis=1)
    # plt.plot(m)
    # plt.show()
    # for i in range(55,60):
    #     plt.plot(m[i, :])
    # plt.plot(m[23,:]) # 58.6 107.8 149.7

    return temp

