import numpy as np
from astropy.io import fits
from tqdm.auto import tqdm
import dask.dataframe as dd
from matplotlib import pyplot as plt
import os
import shutil
import torch
import scipy

train_dir = 'E:\\DLcode\\dr5data\\st\\afgk\\'

del_dir = 'E:\\DLcode\\dr5data\\highdata\\F\\'
del_dir2 = 'E:\\DLcode\\dr5data\\highdata\\AFGK\\'
del_dir3 = 'E:\\DLcode\\dr5data\\highdata\\G\\'
del_dir4 = 'E:\\DLcode\\dr5data\\highdata\\K\\'

target_dir='E:\\DLcode\\dr5data\\randVALIDafgk\\'
target2index = {'O':1, 'B':2,'A':3,'F':4,'G':5,'K':6,'M':7}
train_list = os.listdir(train_dir)
del_list = os.listdir(del_dir)
del_list2 = os.listdir(del_dir2)
del_list3 = os.listdir(del_dir3)
del_list4 = os.listdir(del_dir4)
print('\ntrain: {}  del: {}'.format(len(train_list), len(del_list)))
target_set = set(train_list).difference( set(del_list))
target_set = set(target_set).difference( set(del_list2))
target_set = set(target_set).difference( set(del_list3))
target_set = set(target_set).difference( set(del_list4))
# slist = dd.read_csv('E:\\DLcode\\dr5data\\dr5_stellar.csv', delimiter="|")
# slist = slist.compute()
# slist.set_index('obsid', inplace = True)
print('target: {}'.format(len(target_set )))
for name in tqdm(target_set, mininterval=3):
    o_hdu = fits.open(train_dir + name)
    # if head['obsid'] in slist.index:
    if 1 == 1:
        head = o_hdu[0].header
        # output = prem(o_hdu)
        flux = o_hdu[0].data[0].byteswap().newbyteorder()
        wavelength = o_hdu[0].data[2].byteswap().newbyteorder()
        data = torch.stack((torch.tensor(flux),torch.tensor(wavelength),torch.zeros([len(flux)])),0)
        data[2][0] = target2index[head['subclass'][0]]
        # print(head['subclass\n'])
        # data[2][0] = 1
        data[2][1] = head['z']
        if int(o_hdu[0].header['DATA_V'][-1])>=7:
            data[2][2] = head['FIB_MASK']
        # elif int(o_hdu[0].header['DATA_V'][-1])>=5:
        #     data[2][2] = head['FIBERMAS']
        data[2][3] = float(head['SNRU'])
        data[2][4] = float(head['SNRG'])
        data[2][5] = float(head['SNRR'])
        data[2][6] = float(head['SNRI'])
        data[2][7] = float(head['SNRZ'])
        # data[2][8] = slist.loc[head['obsid'], 'teff']
        # data[2][9] = slist.loc[head['obsid'], 'teff_err']
        data[2][10] = float(head['subclass'][1])
        data[2][11] = head['obsid']
        torch.save(data,'{}valid\\{}.pt'.format(target_dir, name))
        o_hdu.close()
        # os.remove(train_dir + name)
print('\ntrain done\n')