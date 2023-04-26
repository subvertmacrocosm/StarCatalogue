import torch
from astropy.io import fits
import shutil
import os
from tqdm.auto import tqdm
import numpy as np

validdir = 'E:\\DLcode\\dr5data\\st\\StarData\\o\\train\\'
fitsdir = 'E:\\DLcode\\dr5data\\ostar\\'
targetdir = 'E:\\DLcode\\dr5data\\outo\\'
validlist = os.listdir(validdir)
for name in tqdm(validlist,mininterval=3):
    pt = torch.load(validdir + name)
    o_hdu = fits.open(fitsdir + name[:-3])
    pt[2][11] = o_hdu[0].header['obsid']
    torch.save(pt,'{}\\{}.pt'.format(targetdir,name))
    o_hdu.close()
print('done')