
import pyfilterbank.gammatone as gt
import matplotlib.pyplot as plt
import numpy as np


# Press the green button in the gutter to run the script.
# if __name__ == '__main__':


bw_factor_list = [0.05,0.1]
title_lis =['(a) Bandwidth factor=0.05','(b) Bandwidth factor=0.1']

band_width=[5,5000]
start_band = gt.hertz_to_erbscale(band_width[0])
end_band = gt.hertz_to_erbscale(band_width[1])
channels=200

fontsize=8
def _create_impulse(num_samples):
    sig = np.zeros(num_samples) + 0j
    sig[0] = 1.0
    return sig

def plotfun(x, y):
    # ax.semilogx(x, 20 * np.log10(np.abs(y) ** 2))
    axe.plot(x, 20 * np.log10(np.abs(y) ** 2))
    # ax.plot(x, 10 * np.log10(np.abs(y)))

fig = plt.figure(figsize=(6*1.5,6))
for i, band_width_factor in enumerate(bw_factor_list):
    gfb = gt.GammatoneFilterbank(samplerate=10000, bandwidth_factor=band_width_factor,
                                       order=4, startband=start_band, endband=end_band,
                                       density =(end_band-start_band) / (channels-band_width_factor), normfreq=0)
    axe = fig.add_subplot(2,2,3+i)

    gfb.freqz(nfft=2*5000, plotfun=plotfun)
    # axe.set_title(title_lis[i], color='black', y=-0.3,fontsize=fontsize)
    axe.set_xlim([0, 5000])
    axe.set_ylim([-500, 10])
    # axe.set_xlabel('Frequency(Hz)', fontsize=fontsize)
    # axe.set_ylabel('Attenunation(db)', fontsize=fontsize)

    axe = fig.add_subplot(2,2,1+i)
    gfb.freqz(nfft=2*5000, plotfun=plotfun)
    # axe.set_title(title_lis[i], color='black', y=-0.3)
    axe.set_xlim([100, 120])
    axe.set_ylim([-80, 10])
    # axe.set_xlabel('Frequency(Hz)', fontsize=fontsize)
    # axe.set_ylabel('Attenunation(db)', fontsize=fontsize)

fig.subplots_adjust(left=0.083,bottom=0.079,right=0.979,top=0.974,wspace=0.245,hspace=0.2)
fig.show()
filename = '../fig/E1/E1 Bandwidth factor and independence.tif'
fig.savefig(filename,dpi=450)




gt.example_filterbank()
a=10
    # print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
