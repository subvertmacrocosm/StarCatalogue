# import os
# import numpy as np
# import shutil
# from random import sample
# from numpy import random as nrd
# from tqdm.auto import tqdm
#
# filedir = 'F:\\dr5SNR\\GALAXY_SNR\\bad'
# targetdir = ''
# name_list = os.listdir(filedir)
# n = 1000
# choose_list = nrd.random_sample(name_list, n)  # 随机抽选n个
#
# for name in tqdm(name_list, desc = 'pick'):
#     shutil.copy(filedir + name, targetdir + name)
# print('\n------done------')

# import os
# import shutil
#
# import random
#
# filedir = 'F:\\backup\\dr5SNR\\QSO_SNR\\less10\\'
# traindir = 'E:\\DLcode\\dr5data\\SNRdataset\\train\\'
# validdir = 'E:\\DLcode\\dr5data\\QSO_SNR\\QSOvalid\\'
#
# file_list = os.listdir(filedir)
# choose_valid_list = random.sample(file_list, 500)
#
# for name in choose_valid_list:
#     shutil.copy(filedir + name, validdir + name)
#
# print('done')