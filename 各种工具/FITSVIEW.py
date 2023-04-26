import numpy as np
import torch

from astropy.io import fits
from matplotlib import pyplot as plt
import os

def mms(num, min, max):  # min-max归一化
    return (num - min) / (max - min)

def min2(a, b):
    if a>=b:
        return b
    else:
        return a

def max2(a, b):
    if a>=b:
        return a
    else:
        return b
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


roots = 'E:\\DLcode\\dr5data\\dr5_52453\\train\\'
filename = 'spec-56200-EG042015S023742V03_sp01-175.fits'
spec = fits.open(roots + filename)
plt.plot(spec[0].data[2], spec[0].data[0])
plt.title('{}'.format(filename))
plt.show()
min_wave = torch.load('E:\\DLcode\\min_wave.pth')
max_wave = torch.load('E:\\DLcode\\max_wave.pth')
filedir = 'E:\\DLcode\\dr5data\\QSO_high\\'
lokkdir = 'E:\\DLcode\\dr5data\\dr5_52453\\train\\'
pathdir = os.listdir(filedir)
max_wave = torch.load('E:\\DLcode\\max_wave.pth')
min_wave = torch.load('E:\\DLcode\\min_wave.pth')
#pathdir = np.load('E:\\DLcode\\randnet\\select_name.npy')
j=0
for name in pathdir:
    #clss=fits.open(lokkdir + name[0:-4])
    clss=fits.open(filedir + name)
    spec_wave = clss[0].data[2]
    z = clss[0].header['Z']
    cla_obj = clss[0].header['CLASS']
    spec = clss[0].data[0][find_nearest(spec_wave, min_wave)+400:find_nearest(spec_wave, min_wave) + 3200]
    max_min = (np.max(clss[0].data[0])-np.min(clss[0].data[0]))
    err=clss[0].data[1][find_nearest(spec_wave, min_wave)+400:find_nearest(spec_wave, min_wave) + 3200]
    SNRall = np.sum(np.multiply(spec,err**0.5))/len(err)
    if SNRall > 6:
        print('min_wave: {}'.format(clss[0].data[2]))
        print('err:{}'.format(err))
        print('SNRALL:{}'.format(SNRall))
        snr = (clss[0].header['SNRU'] + clss[0].header['SNRG'] + clss[0].header['SNRR'] + clss[0].header['SNRI'] + clss[0].header['SNRZ'])/5
        print('---spec: {}---\nvar: {}\nSNR: {}'.format(name,np.var(clss[0].data[0]), snr))
        print('snr:{} {} {} {} {}'.format(clss[0].header['SNRU'],clss[0].header['SNRG'],clss[0].header['SNRR'],clss[0].header['SNRI'],clss[0].header['SNRZ']))
        print('max-min: {}\n'.format(max_min))
        # print('wave:{}'.format(clss[0].data[2]))
        print('error: {}'.format(clss[0].header['FIBERMAS']))
        print('error: {:0>9}'.format(str(bin(clss[0].header['FIBERMAS']))[2:]))
        fig1 = plt.figure(1)
        plt.ylabel('Flux')
        plt.xlabel('WaveL')
        plt.plot(clss[0].data[2],clss[0].data[0])
        plt.title('CLASS: {}   SNR: {}\nvar: {}\nz: {}'.format(cla_obj,snr, np.var(clss[0].data[0]),z))
        # plt.savefig('E:\\errorUnkown\\squares_plot.png', bbox_inches='tight')
        if z >3:
            plt.show()
        plt.cla()
        j=j+1
        #snr6.append(name[0:-4])

print('done')
print('done,{}'.format(j))