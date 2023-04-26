import numpy as np
from astropy.io import fits
from tqdm.auto import tqdm
import pandas as pd
from matplotlib import pyplot as plt
import os
import shutil
import torch
from redshift import prem

View_dir = 'E:\\DLcode\\dr5data\\highdata\\A\\'


target2index = {'O':1, 'B':2,'A':3,'F':4,'G':5,'K':6,'M':7}
train_list = os.listdir(View_dir)

print('\nView: {}  '.format(len(train_list)))
for name in tqdm(train_list, mininterval=3):
    o_hdu = fits.open(View_dir + name)
    head = o_hdu[0].header
    output = prem(o_hdu)
    data = torch.stack((torch.tensor(output[0]),torch.tensor(output[1]),torch.zeros([len(output[0])])),0)
    # data[2][0] = target2index[head['subclass'][0]]
    print(head['subclass\n'])
    data[2][0] = 1
    data[2][1] = head['z']
    if int(o_hdu[0].header['DATA_V'][-1])>=7:
        data[2][2] = head['FIB_MASK']
    elif int(o_hdu[0].header['DATA_V'][-1])>=5:
        data[2][2] = head['FIBERMAS']
    data[2][3] = float(head['SNRU'])
    data[2][4] = float(head['SNRG'])
    data[2][5] = float(head['SNRR'])
    data[2][6] = float(head['SNRI'])
    data[2][7] = float(head['SNRZ'])


    plt.subplot(2, 1, 1)
    plt.plot(output[0], output[3])
    plt.plot(output[0], output[4])
    plt.title(o_hdu[0].header['subclass'])
    plt.subplot(2, 1, 2)
    plt.plot(output[0], output[1])
    plt.show()
    o_hdu.close()
    plt.close('all')
print('\ntrain done\n')



print('\nvalid done')

