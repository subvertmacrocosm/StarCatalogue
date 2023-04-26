import numpy as np
from astropy.io import fits
from tqdm.auto import tqdm
import dask.dataframe as dd
from matplotlib import pyplot as plt
from redshift import prem
import os
import shutil
import torch
import scipy
# from scipy.optimize import curve_fit as cf

train_dir = 'E:\\DLcode\\dr5data\\randVALIDafgk\\valid\\'
valid_dir = 'E:\\DLcode\\dr5data\\highdata\\A\\'

target_dir = 'E:\\DLcode\\dr5data\\randVALIDafgk\\vpnoredshift\\'
target2index = {'O':1, 'B':2,'A':3,'F':4,'G':5,'K':6,'M':7}
index2target = ['err', 'O', 'B','A','F','G','K','M']
train_list = os.listdir(train_dir)
valid_list = os.listdir(valid_dir)
print('\ntrain: {}  valid: {}'.format(len(train_list), len(valid_list)))
def plk(x, t, b,c):
    c1 = 3.7418*(10**-30)
    c2 = 0.014388
    x=x*(10**(-10))

    return (c1 / x ** 5) / (np.exp(c2 / (x * t)) - 1)*c + b



param_bounds=([1000,-20000,0],[30000,20000,200000])
for name in tqdm(train_list, mininterval=10):
    o_hdu = torch.load(train_dir + name)
    rs = 1
    if rs == 1:
        z = o_hdu[2][1]
        z=0
        if z == -9999:
            z = 0
            print('\nz -9999 {}\n')

        wavelength = o_hdu[1]
        flux = o_hdu[0]
        doppler_factor = 1 / (z + 1)
        new_wavelength = wavelength * doppler_factor
        target_wavelength = np.arange(new_wavelength.min() + 1, new_wavelength.max() - 1, 1)
        new_flux = np.interp(target_wavelength, new_wavelength, flux)

        kernel_size = 499
        sc = scipy.signal.medfilt(new_flux, kernel_size)

        outde = np.divide(new_flux[kernel_size // 4:len(sc) - kernel_size // 4],
                          sc[kernel_size // 4:len(sc) - kernel_size // 4])
        outde[outde > 100] = 1
        outde[outde > 5] = 5
        outde[outde < -1] = 1

        ot1 = outde
        ot2 = target_wavelength[kernel_size // 4:len(sc) - kernel_size // 4]
        # flux_cut = sc[500:]
        # wave_cut = target_wavelength[500:]
        # popt, pcov = cf(plk, wave_cut, flux_cut, bounds=param_bounds)
        # tfit = popt[0]


    data = torch.stack((torch.tensor(ot1),torch.tensor(ot2),torch.zeros([len(ot1)])),0)
    data[2][0:20] = o_hdu[2][0:20]
    # data[2][21] = float(tfit)
    # print(head['subclass\n'])
    # data[2][0] = 1


    # elif int(o_hdu[0].header['DATA_V'][-1])>=5:
    #     data[2][2] = head['FIBERMAS']
    # plt.subplot(2,1,1)
    # plt.title('{}{} T: {} Tf: {}'.format(index2target[int(data[2][0])], int(data[2][10]), data[2][8], tfit))
    # plt.plot(wave_cut, flux_cut)
    # plt.plot(wave_cut, plk(wave_cut,*popt))
    # plt.subplot(2, 1, 2)
    # plt.plot(data[1],data[0])
    # plt.show()
    torch.save(data,'{}{}.pt'.format(target_dir, name))
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

