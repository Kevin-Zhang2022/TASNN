from cla.Layer import AuditoryNerve,Bushy,InferiorColliculus,AudioCortex,Cochlear,InnerHairCell
import numpy as np
import matplotlib.pyplot as plt

scale_range_list= [[0.2,2],[0.2,3]]

fig = plt.figure(figsize=(6,3))
fontsize=8
# title_list=['(a)','(b)','(c)','(d)']

for i in range(1,3):
    scale_range = scale_range_list[i-1]
    bushy = Bushy(2000, 500, scale_range=scale_range)  # 0.5/2
    mask = np.array(bushy.mask)
    print(bushy.of)
    axe = fig.add_subplot(1, 2, i)
    axe.imshow(mask, aspect='auto')
    fig.show()

    axe.set_xlim([980,1010])
    axe.set_ylim([370, 367])
    axe.set_yticks([367,368,369,370])

    # axe.set_xlabel('Input channel ordinal', fontsize=fontsize)
    # axe.set_ylabel('Output channel ordinal', fontsize=fontsize)

fig.subplots_adjust(left=0.095,bottom=0.15,right=0.965,top=0.975,wspace=0.329,hspace=0.338)
fig.show()
filename = '../fig/E2/E2 SD range and independence.tif'
fig.savefig(filename,dpi=450)

# plt.imshow(mask, aspect='auto')
# sum = 0
# for i in range(500 - 1):
#     sum += np.sum(mask[i, :] * mask[i + 1, :])
# over_lapping_coe = sum / out_features
# print(over_lapping_coe)


a=10