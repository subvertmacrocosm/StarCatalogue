import os
import shutil
import numpy as np
from astropy.io import fits
from tqdm.auto import tqdm



filedir = 'E:\\DLcode\\dr5data\\dr5_52453\\train\\'
STARdir = 'F:\\backup\\dr5SNR\\STAR_SNR\\'
GALAXYdir = 'F:\\backup\\dr5SNR\\GALAXY_SNR\\'
QSOdir = 'F:\\backup\\dr5SNR\\QSO_SNR\\'
Unknowndir = 'F:\\backup\\dr5SNR\\Unknown_SNR\\'

target2dir = {'STAR': STARdir, 'GALAXY': GALAXYdir, 'QSO': QSOdir, 'Unknown': Unknowndir}

pathdir = os.listdir(filedir)

for name in tqdm(pathdir, desc = 'pick'):
    clss=fits.open(filedir + '\\' + name)
    targetdir = target2dir[clss[0].header['CLASS']]
    if clss[0].header['FIBERMAS'] == 0 and  (np.max(clss[0].data[0])-np.min(clss[0].data[0]))<10000:
        SNR = (clss[0].header['SNRU']+clss[0].header['SNRG']+clss[0].header['SNRR']+clss[0].header['SNRI']+clss[0].header['SNRZ'])/5
        if SNR < 10:
            shutil.copy(filedir + name, targetdir + 'less10\\' + name)
        elif SNR < 20:
            shutil.copy(filedir + name, targetdir + '10\\' + name)
        elif SNR < 30:
            shutil.copy(filedir + name, targetdir + '20\\' + name)
        elif SNR < 40:
            shutil.copy(filedir + name, targetdir + '30\\' + name)
        elif SNR < 50:
            shutil.copy(filedir + name, targetdir + '40\\' + name)
        elif SNR < 60:
            shutil.copy(filedir + name, targetdir + '50\\' + name)
        else:
            shutil.copy(filedir + name, targetdir + 'more60\\' + name)
    else:
        shutil.copy(filedir + name, targetdir + 'bad\\' + name)
    clss.close()
print("done")
