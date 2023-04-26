import numpy as np
import torch

from astropy.io import fits
from matplotlib import pyplot as plt
import os

import scipy.signal

# need 3900-5800\,\AA\,and 8450--8950\,\
def prem(o_hdu):
    z = o_hdu[0].header['Z']
    z=0
    if z == -9999:
        z = 0
        print('\nz -9999 {}\n'.format(o_hdu[0].header['subclass']))
    if int(o_hdu[0].header['DATA_V'][-1])==9:
        wavelength = o_hdu[1].data[0][2]
        flux = o_hdu[1].data[0][0]
        doppler_factor = 1 / (z + 1)
        new_wavelength = wavelength * doppler_factor
        target_wavelength= np.arange(np.min(new_wavelength)+1,np.max(new_wavelength)-1,1)
        new_flux = np.interp(target_wavelength, new_wavelength, flux)
    else:
        wavelength = o_hdu[0].data[2]
        flux = o_hdu[0].data[0]
        doppler_factor = 1 / (z + 1)
        new_wavelength = wavelength * doppler_factor
        target_wavelength= np.arange(np.min(new_wavelength)+1,np.max(new_wavelength)-1,1)
        new_flux = np.interp(target_wavelength, new_wavelength, flux)



    kernel_size = 699
    sc = scipy.signal.medfilt(new_flux, kernel_size)







    outde = np.divide(new_flux[kernel_size//4:len(sc) - kernel_size//4], sc[kernel_size//4:len(sc) - kernel_size//4])
    outde[outde > 100] = 1
    outde[outde > 10] = 10
    outde[outde < -1] = 1



    return target_wavelength[kernel_size//4:len(sc) - kernel_size//4], outde

    # return target_wavelength[kernel_size//4:len(sc) - kernel_size//4], outde, kernel_size, new_flux[kernel_size//4:len(sc) - kernel_size//4], sc[kernel_size//4:len(sc) - kernel_size//4]