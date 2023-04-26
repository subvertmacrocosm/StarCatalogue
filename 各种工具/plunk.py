import numpy as np

from astropy.io import fits
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit as cf
import os
import scipy.signal as signal


def plk(x, t, b,c):
    c1 = 3.7418*(10**-30)
    c2 = 0.014388
    x=x*(10**(-10))

    return (c1 / x ** 5) / (np.exp(c2 / (x * t)) - 1)*c + b


param_bounds=([1000,-10000,0],[1000000,2000,200000])
# param_bounds=([-np.inf,-np.inf],[np.inf,np.inf])
specdir = 'E:\\DLcode\\dr5data\\st\\f\\'
for name in os.listdir(specdir):
    o_hdu = fits.open(specdir + name)
    if int(o_hdu[0].header['DATA_V'][-1]) >=8:
        head = o_hdu[0].header
        data  = o_hdu[1].data[0]
    else:
        head = o_hdu[0].header
        data  = o_hdu[0].data
    flux = data[0]
    wave_length = data[2]
    z = float(head['z'])
    if z == -9999:
        z = 0
    doppler_factor = 1 / (z + 1)
    new_wavelength = wave_length * doppler_factor
    target_wavelength = np.arange(np.min(new_wavelength) + 1, np.max(new_wavelength) - 1, 1)
    new_flux = np.interp(target_wavelength, new_wavelength, flux)
    flux_cut = new_flux[800:len(new_flux)-500]
    wave_cut = target_wavelength[800:len(new_flux)-500]
    k_size = 199
    x = np.arange(1000,9000,5)

    sc = signal.medfilt(new_flux,k_size)[800:len(new_flux)-500]
    popt, pcov = cf(plk,wave_cut,sc,bounds=param_bounds)

    sc2 = signal.medfilt( flux_cut, 199)
    popt2, pcov2 = cf(plk, wave_cut, sc2, bounds=param_bounds)
    y = plk(x,*popt)
    y2 = plk(x, *popt2)

    mean = np.mean(sc)  # 1.y mean
    ss_tot = np.sum((sc - mean) ** 2)  # 2.total sum of squares
    ss_res = np.sum((sc - plk(wave_cut, *popt)) ** 2)  # 3.residual sum of squares
    r_squared = 1 - (ss_res / ss_tot)  # 4.r squared

    mean2 = np.mean(sc2)  # 1.y mean
    ss_tot2 = np.sum((sc2 - mean2) ** 2)  # 2.total sum of squares
    ss_res2 = np.sum((sc2 - plk(wave_cut, *popt2)) ** 2)  # 3.residual sum of squares
    r_squared2 = 1 - (ss_res2 / ss_tot2)  # 4.r squared
    print(popt)
    print(head['subclass'])
    print('\nwave: {}'.format(wave_cut.min()))
    pl_flux = plk(target_wavelength,*popt)
    plt.title(head['subclass']+'\n'+str(head['z']))
    plt.plot(target_wavelength,new_flux)
    plt.plot(x,y,label='T: %5.3f, b: %5.3f, c: %5.3f\n R^2: {}'.format(r_squared) %tuple(popt))
    plt.plot(x,y2,label='222 - T: %5.3f, b: %5.3f, c: %5.3f\n R^2: {}'.format(r_squared2) %tuple(popt))
    # plt.plot(wave_cut, sc,label='med f')
    plt.legend()

    plt.show()

    plt.subplot(2, 1, 1)
    yy = np.divide(flux_cut,plk(wave_cut,*popt))
    plt.plot(wave_cut,yy)
    plt.subplot(2, 1, 2)
    ff = np.divide(flux_cut,plk(wave_cut,*popt2))
    plt.plot(wave_cut,ff)
    plt.show()
    plt.close('all')
    print('done')
