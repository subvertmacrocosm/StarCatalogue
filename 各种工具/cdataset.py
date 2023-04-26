import numpy as np
from astropy.io import fits
import torch
import os
import shutil
from tqdm.auto import tqdm
import random

filedir = 'F:\\dr5pth\\QSO\\'
traindir = 'E:\\DLcode\\dr5data\\randDataset\\train\\'
validdir = 'E:\\DLcode\\dr5data\\randDataset\\valid\\'
file_list = os.listdir(filedir)
valid_list = random.sample(file_list, 1000)
train_list = list(set(file_list)-set(valid_list))
print('train: {}\nvalid: {}'.format(len(train_list), len(valid_list)))

for name in tqdm(train_list, mininterval=1):
    shutil.copy(filedir + name, traindir + name)
print('train dataset done')
for name in tqdm(valid_list, mininterval=1):
    shutil.copy(filedir + name, validdir + name)
print('valid dataset done')