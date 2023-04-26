import numpy as np
import torch

from astropy.io import fits
from matplotlib import pyplot as plt
import os
from tqdm.auto import tqdm
import scipy.signal

# need 3900-5800\,\AA\,and 8450--8950\,\
o_hdu = fits.open('E:\\DLcode\\dr5data\\aa\\spec-55966-B5596605_sp08-031.fits')
z = o_hdu[0].header['Z']
target2index = {'O':1, 'B':2,'A':3,'F':4,'G':5,'K':6,'M':7}
for c_z in tqdm(np.arange(-0.01,0.01,0.00001),mininterval=2):
    if z == -9999:
        z = 0
        print('\nz -9999 {}\n'.format(o_hdu[0].header['subclass']))
    if int(o_hdu[0].header['DATA_V'][-1])==9:
        wavelength = o_hdu[1].data[0][2]
        flux = o_hdu[1].data[0][0]
        doppler_factor = 1 / (z + 1)
        new_wavelength = wavelength * doppler_factor
        new_wavelength = new_wavelength + new_wavelength * c_z
        target_wavelength= np.arange(np.min(new_wavelength)+1,np.max(new_wavelength)-1,1)
        new_flux = np.interp(target_wavelength, new_wavelength, flux)
    else:
        wavelength = o_hdu[0].data[2]
        flux = o_hdu[0].data[0]
        doppler_factor = 1 / (z + 1)
        new_wavelength = wavelength * doppler_factor
        new_wavelength = new_wavelength + new_wavelength * c_z
        target_wavelength= np.arange(np.min(new_wavelength)+1,np.max(new_wavelength)-1,1)
        new_flux = np.interp(target_wavelength, new_wavelength, flux)

    kernel_size = 699
    sc = scipy.signal.medfilt(new_flux, kernel_size)







    outde = np.divide(new_flux[kernel_size//4:len(sc) - kernel_size//4], sc[kernel_size//4:len(sc) - kernel_size//4])
    outde[outde > 100] = 1
    outde[outde > 10] = 10
    outde[outde < -1] = 1
    # plt.plot(target_wavelength[kernel_size//4:len(sc) - kernel_size//4], outde)
    # plt.show()
    output = [target_wavelength[kernel_size//4:len(sc) - kernel_size//4], outde]
    data = torch.stack((torch.tensor(output[0]), torch.tensor(output[1]), torch.zeros([len(output[0])])), 0)
    data[2][0] = target2index[o_hdu[0].header['subclass'][0]]
    data[2][1] = c_z
    torch.save(data,'E:\\DLcode\\Z\\z_{}.pt'.format(c_z))

