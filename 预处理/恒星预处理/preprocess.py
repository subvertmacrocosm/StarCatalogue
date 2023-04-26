import numpy as np
from astropy.io import fits
from tqdm.auto import tqdm
import dask.dataframe as dd
from matplotlib import pyplot as plt
from redshift import prem
import os
import shutil
import torch

train_dir = 'E:\\DLcode\\dr5data\\st\\afgk\\'
valid_dir = 'E:\\DLcode\\dr5data\\highdata\\K\\'

target_dir = 'E:\\DLcode\\dr5data\\highset\\'
target2index = {'O':1, 'B':2,'A':3,'F':4,'G':5,'K':6,'M':7}
train_list = os.listdir(train_dir)
valid_list = os.listdir(valid_dir)
print('\ntrain: {}  valid: {}'.format(len(train_list), len(valid_list)))
slist = dd.read_csv('E:\\DLcode\\dr5data\\dr5_stellar.csv', delimiter="|")
slist = slist.compute()
slist.set_index('obsid', inplace = True)
for name in tqdm(train_list, mininterval=3):
    o_hdu = fits.open(train_dir + name)

    # test
    # ohdu = fits.open('E:\\DLcode\\dr5data\\spec-57815-HD160238N263643M02_sp15-062.fits')

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
    data[2][8] = slist.loc[head['obsid'], 'teff']
    data[2][9] = slist.loc[head['obsid'], 'teff_err']
    data[2][10] = float(head['subclass'][1])
    data[2][11] = head['obsid']
    torch.save(data,'{}train\\{}.pt'.format(target_dir, name))
    o_hdu.close()
    # os.remove(train_dir + name)
print('\ntrain done\n')
# for name in tqdm(valid_list, mininterval=3):
#     hdua = fits.open(valid_dir + name)
#     hdu = hdua[0]
#     head = hdu.header
#     output = pre_main(hdu)
#     data = torch.stack((torch.tensor(output[0]),torch.tensor(output[1]),torch.zeros([len(output[0])])),0)
#     data[2][0] = target2index[head['subclass'][0]]
#     data[2][1] = head['z']
#     data[2][2] = head['FIBERMAS']
#     data[2][3] = head['SNRU']
#     data[2][4] = head['SNRG']
#     data[2][5] = head['SNRR']
#     data[2][6] = head['SNRI']
#     data[2][7] = head['SNRZ']
#     torch.save(data,'{}valid\\{}.pt'.format(target_dir, name))
#     hdua.close()
#     os.remove(valid_dir + name)


print('\nvalid done')

