import numpy as np
from astropy.io import fits
import torch
import os
import shutil
from tqdm.auto import tqdm
import random

# 查看四分类数据集
traindir = 'E:\\DLcode\\dr5data\\testvalid\\randDataset\\randDataset\\train\\'
STAR = []
GALAXY = []
QSO = []
Unknown = []
filelist = os.listdir(traindir)
for name in tqdm(filelist,mininterval=0.5):
    spec = torch.load(traindir + name)
    if spec[0][0].int() == 0:
        STAR.append(name)
    elif spec[0][0].int() == 1:
        GALAXY.append(name)
    elif spec[0][0].int() == 2:
        QSO.append(name)
    else:
        Unknown.append(name)
print('done')
print('{} \n{} {} {} {}'.format(len(filelist),len(STAR),len(GALAXY),len(QSO),len(Unknown)))
validpath = 'E:\\DLcode\\dr5data\\testvalid\\randDataset\\randDataset\\valid\\'
validlist = os.listdir(validpath)
choose=[]
# STAR = list(set(STAR) - set(validlist))
# QSO = list(set(QSO) - set(validlist))
# GALAXY = list(set(GALAXY) - set(validlist))
choose = list(choose + random.sample(STAR, 11453))
choose = list(choose + random.sample(QSO, 11453))
choose = list(choose + random.sample(GALAXY, 11453))
for name in tqdm(choose):
    os.remove(traindir + name)

print('done')
# 查看恒星数据集
#
# traindir = 'E:\\DLcode\\dr5data\\dr5_52453\\train\\'
# A=set()
# B=set()
# F=set()
# G=set()
# K=set()
# M=set()
# O=set()
# un = 0
# filelist = os.listdir(traindir)
# for name in tqdm(filelist,mininterval=0.5):
#      spec = fits.open(traindir + name)
#      if spec[0].header['CLASS'] == 'STAR':
#          subc = spec[0].header['SUBCLASS']
#          if subc[0] == 'A':
#              A.add(name)
#          elif subc[0] == 'B':
#              B.add(name)
#          elif subc[0] == 'F':
#              F.add(name)
#          elif subc[0] == 'G':
#              G.add(name)
#          elif subc[0] == 'K':
#              K.add(name)
#          elif subc[0] == 'M':
#              M.add(name)
#          elif subc[0] == 'O':
#              O.add(name)
#          else:
#              un = un +1
#
# print('done')
# print('A {}, B {}, F {}, G {}, K {}, M {}, O {}'.format(len(A),len(B),len(F),len(G),len(K),len(M),len(O)))