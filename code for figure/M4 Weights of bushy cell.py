from cla.Layer import AuditoryNerve,Bushy,InferiorColliculus,AudioCortex,Cochlear,InnerHairCell
import numpy as np
import matplotlib.pyplot as plt

scale_range=[0.5,2]

fig = plt.figure(figsize=(9,6))
fontsize=8
# title_list=['(b)','(c)','(d)']
y1 =-0.12
y2 =-0.27

# bbox = {"facecolor": "red", "alpha": 0.5}
# styles = {"size": 10, "color": "black", "bbox": bbox}
# fig.subplots_adjust(left=0.08,bottom=0.121,right=0.964,top=0.975,wspace=0.5,hspace=0.338)

bushy = Bushy(2000, 500, scale_range=scale_range)  # 0.5/2
mask = np.array(bushy.mask)
axe0 = fig.add_subplot(1, 3, 1)
image = axe0.imshow(mask, aspect='auto')
# cax = fig.add_axes([axe.get_position().x1+0.01,axe.get_position().y0,
#                     0.02,axe.get_position().height])
#
# cbar = fig.colorbar(image,cax=cax)
# # 设置颜色条标签
# cbar.set_label('Color bar of weights',fontsize=fontsize,loc='left')
# plt.show()

# fig.colorbar(axe, cax=cax) # Similar to fig.colorbar(im, cax = cax)
#
#
# plt.colorbar(axe,)


# x=[100,250,500,1000]
# y = result[(i-1)*4:i*4,0]
# axe.plot(x,y)
# axe0.set_title('(a) Weights of bushy layer', color='black',y=y1,fontsize=fontsize)
axe0.set_xlim([0,2001])
axe0.set_ylim([499, 0,])
# axe0.set_xlabel('Input channel ordinal', fontsize=fontsize)
# axe0.set_ylabel('Output channel ordinal', fontsize=fontsize)
fig.show()

axe = fig.add_subplot(2, 3, 2)
axe.imshow(mask[12:17,0:40],aspect='auto')

# axe.set_xlim([2,11])
# axe.set_ylim([4,0])
# axe.set_xlabel('Input channel ordinal', fontsize=fontsize)
# axe.set_ylabel('Output channel ordinal', fontsize=fontsize)
# axe.set_xticks([2,5,10])
# axe.set_title('(b) Weights for low frequency ', color='black',y=y2,fontsize=fontsize)
axe.set_xticks([0,20,39])
axe.set_xticklabels([0,20,39])
axe.set_yticks([0,2,4])
axe.set_yticklabels([12,14,16])

axe = fig.add_subplot(2, 3, 5)
axe.imshow(mask[450:452,1530:1570],aspect='auto')
# axe.set_title('(c) Weights for high frequency ', color='black',y=y2,fontsize=fontsize)
axe.set_xticks([0,20,39])
axe.set_xticklabels([1530,1550,1539])
axe.set_yticks([0,1])
axe.set_yticklabels([450,451])
# axe.set_xlabel('Input channel ordinal', fontsize=fontsize)
# axe.set_ylabel('Output channel ordinal', fontsize=fontsize)


axe = fig.add_subplot(2, 3, 3)
axe.plot(mask[15,0:100])
# axe.set_title('(d) Normal distribution for output channel 15', color='black',y=y2,fontsize=fontsize)
axe.set_xlim([0,100])
axe.set_ylim([0, 1])
# axe.set_xlabel('Input channel ordinal', fontsize=fontsize)
# axe.set_ylabel('Mask value', fontsize=fontsize)
# axe.text(55,0.8,s="Mean=16.17\nSD=0.56",ha='left',fontsize=fontsize+3)

# print(bushy.mean[15])
# print(bushy.scale[15])

axe = fig.add_subplot(2, 3, 6)
axe.plot(mask[450,:])
# axe.set_title('(e) Normal distribution for output channel 450', color='black',y=y2,fontsize=fontsize)
axe.set_xlim([1500,1600])
axe.set_ylim([0, 1])
# axe.set_xlabel('Input channel ordinal', fontsize=fontsize)
# axe.set_ylabel('Mask value', fontsize=fontsize)
# axe.text(1555,0.8,s="Mean=1546.63\nSD=1.87",ha='left',fontsize=fontsize)
# print(bushy.mean[450])
# print(bushy.scale[450])

fig.subplots_adjust(left=0.08,bottom=0.121,right=0.864,top=0.975,wspace=0.293,hspace=0.338)

cax = fig.add_axes([axe.get_position().x1+0.04,axe.get_position().y0,
                    0.02,axe0.get_position().height])
cbar = fig.colorbar(image,cax=cax)
# 设置颜色条标签
# cbar.set_label('Color bar of weights',fontsize=fontsize,loc='center')



filename = '../fig/M4 Bushy cell/M4 Weights of bushy cell.tif'
fig.savefig(filename)

# plt.imshow(mask, aspect='auto')
# sum = 0
# for i in range(500 - 1):
#     sum += np.sum(mask[i, :] * mask[i + 1, :])
# over_lapping_coe = sum / out_features
# print(over_lapping_coe)


a=10