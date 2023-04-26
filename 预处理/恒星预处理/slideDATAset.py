import numpy as np
from astropy.io import fits
import torch
import os
import shutil
from tqdm.auto import tqdm
import random

filedir = 'E:\\DLcode\\dr5data\\highnors\\train\\K\\'
traindir = 'E:\\DLcode\\dr5data\\temdataset\\train\\'
validdir = 'E:\\DLcode\\dr5data\\temdataset\\valid\\'
file_list = os.listdir(filedir)
valid_list = random.sample(file_list, 5000)
train_list = list(set(file_list)-set(valid_list))
print('train: {}\nvalid: {}'.format(len(train_list), len(valid_list)))

for name in tqdm(train_list, mininterval=3):
    shutil.copy(filedir + name, traindir + name)
print('train dataset done')
for name in tqdm(valid_list, mininterval=3):
    shutil.copy(filedir + name, validdir + name)
print('valid dataset done')