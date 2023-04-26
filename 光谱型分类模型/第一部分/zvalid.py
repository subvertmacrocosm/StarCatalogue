import os
import time

from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from astropy.io import fits
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from model import StarNet
from sklearn import preprocessing
from tqdm.auto import tqdm

target2index = {'STAR': 0, 'GALAXY': 1, 'QSO': 2, 'Unknown': 3}
device = torch.device("cpu")
if torch.cuda.is_available() == 1:
    device = torch.device("cuda")

start_time = time.time()


def mms(num, min, max):
    return ((num - min) / (max - min))




class StarDataset(Dataset):
    def __init__(self, root):
        self.root = root

        self.set = []

        for name in tqdm(os.listdir(self.root),mininterval=5):
            spec1 = torch.load(root + name)
            target = spec1[2][0].type(torch.int64) - 1
            spec_blue = spec1[1][(spec1[0] > 3900) & (spec1[0] <= 5800)].float()
            spec_red = spec1[1][(spec1[0] >= 8450) & (spec1[0] < 8950)].float()
            spec_blue = torch.nan_to_num(spec_blue,nan=1.0, posinf=1.0, neginf=1.0)
            spec_red = torch.nan_to_num(spec_red, nan=1.0, posinf=1.0, neginf=1.0)
            if len(spec_blue)<1900:
                spec_blue = torch.cat((torch.ones(1900-len(spec_blue)),spec_blue), dim=0)
            if len(spec_red)<500:
                spec_red = torch.cat((torch.ones(500 - len(spec_red)), spec_red), dim=0)
            spec_blue = mms(spec_blue, spec_blue.min(), spec_blue.max()).unsqueeze(0)
            spec_red = mms(spec_red, spec_red.min(), spec_red.max()).unsqueeze(0)
            spec_red = torch.nan_to_num(spec_red, nan=1.0, posinf=1.0, neginf=1.0)
            z = spec1[2][1]
            self.set.append({'spec_blue': spec_blue, 'spec_red': spec_red, 'target': target, 'z':z})



    def __getitem__(self, index):
        spec_blue = self.set[index]['spec_blue']
        spec_red = self.set[index]['spec_red']
        target = self.set[index]['target']
        z = self.set[index]['z']
        return spec_blue, spec_red, target, z

    def __len__(self):
        return len(self.set)



valid_root = 'E:\\DLcode\\Z\\'
# o_valid_root='E:\\DLcode\\dr5data\\st\\StarData\\o\\valid\\'
batch = 256


valid_dataset = StarDataset(root=valid_root)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch)

print("start")

zrange = np.arange(-0.01,0.01,0.00001)
z_matrix = torch.zeros(len(zrange))
z_matrix = z_matrix.to(device)


net = torch.load('E:\\DLcode\\STARnet\\STARnet\\star_model.pth')
net = net.to(device)
net.load_state_dict(torch.load('E:\\DLcode\\STARnet\\STARnet\\star_model_304_0.8616306781768799.pth'))


def zzz_matrix(output, targets, z_matrix, z):
    output = torch.max(output, 1)[1]
    for out, tar, zz in zip(output, target, z):
        if out == tar:
            z_matrix[np.where(zrange == float(zz))] += 1
    return z_matrix


net.eval()

with torch.no_grad():
    for spec_blue,spec_red, target, z in valid_dataloader:
        spec_blue = spec_blue.to(device)
        spec_red = spec_red.to(device)
        target = target.to(device)
        output = net(spec_blue, spec_red)
        z = z.to(device)
        z_matrix = zzz_matrix(output, target, z_matrix, z)

z_matrix = np.array(z_matrix.cpu())
labels = ['O','B','A','F','G','K','M']

plt.plot(zrange,z_matrix)



plt.show()
print('done')