import numpy as np
from torch.utils.data import Dataset

from macls.data_utils.audio import AudioSegment
from macls.data_utils.augmentor.augmentation import AugmentationPipeline
from macls.utils.logger import setup_logger
import math
import os
import sys
import csv
import shutil

file_names = os.listdir('fold1')
file_names_split = []
for i in range(10):
    if os.path.exists(f'{i:d}'):
        pass
    else:
        os.makedirs(f'{i:d}')

for filename in file_names:
    temp = filename.split('-')
    # temp1 = temp[1].split('.')
    shutil.copy('fold1/'+filename,f'{int(temp[1]):d}')



# filt_splited = os.split(file_names,'-')
    # dirs = os.listdir(path2audio)
    # path_split = path2audio.split('/')
    # head = ['class_name', 'class_id']
    #
    # dir_lv2 = os.path.join(path_split[0], path_split[1], path_split[2]).replace('\\', '/')
    #
    # f_label = open(os.path.join(dir_lv2, 'label_list.csv'), 'w', encoding='utf-8', newline='')
    # f_all = open(os.path.join(dir_lv2, 'all_list.csv'), 'w', encoding='utf-8', newline='')
    # f_train = open(os.path.join(dir_lv2, 'train_list.csv'), 'w', encoding='utf-8', newline='')
    # f_test = open(os.path.join(dir_lv2, 'test_list.csv'), 'w', encoding='utf-8', newline='')
    #
    # # this is label list
    # writer_label = csv.writer(f_label)
    # writer_label.writerow(head)
    # class_dic = {}
    # for dir_id, dir in enumerate(dirs):
    #     writer_label.writerow([dir, dir_id])
    #     if dir not in class_dic.keys():
    #         class_dic[dir] = dir_id
    # f_label.close()
    #
    # # now let's do training list
    # writer_all = csv.writer(f_all)
    # path = path2audio
    # if not os.path.exists(path):
    #     # print("error: folder \"", path, "\" does not exit!")
    #     print('error: folder \"', path, '\" does not exit!')
    #     sys.exit()
    #
    # writer_all.writerow(['path', 'class_id'])
    # for root, dirs, files in os.walk(path):
    #     for file in files:
    #         path = os.path.join(root, file).replace('\\', '/')
    #         temp = path.split('/')
    #         class_id = class_dic[temp[-2]]
    #         writer_all.writerow([path, class_id])
    #         # print(path,'\t',class_id)
    # f_all.close()
    #
    # f_all = open(os.path.join(dir_lv2, 'all_list.csv'), 'r', encoding='utf-8', newline='')
    # lines = f_all.readlines()
    # f_all.close()
    #
    # index = np.arange(len(lines) - 1)
    # np.random.shuffle(index)
    #
    # train_index = index[0:int(0.8 * (len(lines) - 1))]
    # test_index = index[int(0.8 * (len(lines) - 1)):]
    #
    # writer_train = csv.writer(f_train)
    # writer_train.writerow(head)
    # for i in range(len(train_index)):
    #     temp = lines[train_index[i] + 1].split(',')
    #     writer_train.writerow([temp[0], temp[1].replace('\r\n', '')])
    # f_train.close()
    #
    # writer_test = csv.writer(f_test)
    # writer_test.writerow(head)
    # for i in range(len(test_index)):
    #     temp = lines[test_index[i] + 1].split(',')
    #     writer_test.writerow([temp[0], temp[1].replace('\r\n', '')])
    # f_test.close()
