import numpy as np
from astropy.io import fits
import torch
import os
import shutil
from tqdm.auto import tqdm

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

target2index = {'STAR': 0, 'GALAXY': 1, 'QSO': 2, 'Unknown': 3}

max_wave = torch.load('E:\\DLcode\\max_wave.pth')
min_wave = torch.load('E:\\DLcode\\min_wave.pth')
root='E:\\DLcode\\dr5data\\dr5_52453\\train\\'
targetpath = 'F:\\dr5pth\\'
filedir = os.listdir(root)
for name in tqdm(filedir):
    spec_fits = fits.open(root + name)
    spec_wave = spec_fits[0].data[2]
    f_header = spec_fits[0].header
    idx_min_max = [find_nearest(spec_wave, min_wave) + 400, find_nearest(spec_wave, min_wave) + 3600]
    spec = spec_fits[0].data[0][idx_min_max[0]: idx_min_max[1]]
    if spec.min() < 0 :
        a = 0
        print('err<0')
    err = spec_fits[0].data[1][idx_min_max[0]: idx_min_max[1]]
    spec_wave = spec_wave[idx_min_max[0]: idx_min_max[1]]
    SNR = np.sum(np.multiply(spec, err**0.5))/len(spec)
    label = torch.zeros(3200)
    label[0] = target2index[spec_fits[0].header['CLASS']]
    label[1] = SNR
    label[2] = spec_fits[0].header['FIBERMAS']
    spec = torch.tensor(spec.byteswap().newbyteorder())
    err = torch.tensor(err.byteswap().newbyteorder())
    spec_wave = torch.tensor(spec_wave.byteswap().newbyteorder())
    output = torch.stack((label, spec, err, spec_wave))

    # torch.save(f_header, '{}.header.pth'.format(targetpath + spec_fits[0].header['CLASS'] + '\\' + name))
    # torch.save(output, '{}.pth'.format(targetpath + spec_fits[0].header['CLASS'] + '\\' + name))
print('done')