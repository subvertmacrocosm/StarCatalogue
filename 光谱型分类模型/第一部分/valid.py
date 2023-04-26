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
import pandas as pd
target2index = {'STAR': 0, 'GALAXY': 1, 'QSO': 2, 'Unknown': 3}
device = torch.device("cpu")
if torch.cuda.is_available() == 1:
    device = torch.device("cuda")

start_time = time.time()


def mms(num, min, max):
    return ((num - min) / (max - min))




class StarDataset(Dataset):
    def __init__(self, root, root2):
        self.root = root
        self.root2 = root2
        self.set = []

        for name in tqdm(os.listdir(self.root),mininterval=5):
            spec1 = torch.load(root + name)
            target = spec1[2][0].type(torch.int64) - 1
            aa=1
            # if target>=2 and target<=5:
            #     aa = np.random.rand(1)
            if aa >=0.75:
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
                obsid = spec1[2][11]

                self.set.append({'spec_blue': spec_blue, 'spec_red': spec_red, 'target': target,'obsid':obsid})

        for name in os.listdir(self.root2):
            spec1 = torch.load(root2 + name)
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
            obsid = spec1[2][11]
            self.set.append({'spec_blue': spec_blue, 'spec_red': spec_red, 'target': target,'obsid':obsid})

    def __getitem__(self, index):
        spec_blue = self.set[index]['spec_blue']
        spec_red = self.set[index]['spec_red']
        target = self.set[index]['target']
        obsid = self.set[index]['obsid']
        return spec_blue, spec_red, target, obsid

    def __len__(self):
        return len(self.set)



valid_root = 'E:\\DLcode\\dr5data\\out1\\'
o_valid_root='E:\\DLcode\\dr5data\\outo\\'
batch = 3


valid_dataset = StarDataset(root=valid_root,root2=o_valid_root)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch)
valid_data_size = len(valid_dataset)
print("start,size:{}".format(valid_data_size))

types_num = 3
conf_matrix = torch.zeros(types_num, types_num)
conf_matrix = conf_matrix.to(device)


net = torch.load('E:\\DLcode\\STARnet\\STARnet\\star_model.pth')
net = net.to(device)
net.load_state_dict(torch.load('E:\\DLcode\\STARnet\\STARnet\\star_model_304_0.8616306781768799.pth'))


def confusion_matrix(output, targets, conf_matrix):
    for out, tar in zip(output, target):
        # if (out == 0) & (output[i][0] < 0.9999): #
        #     out = 3
        if out == 1:
            out =0
        elif out >=2 and out <= 5:
            out = 1
        elif out == 6:
            out = 2
        if tar == 1:
            tar = 0
        elif tar >=2 and tar <= 5:
            tar = 1
        elif tar == 6:
            tar = 2

        conf_matrix[out, tar] += 1
    return conf_matrix


net.eval()
valid_step = 0
output_list = torch.zeros(valid_data_size)
obsid_list = torch.zeros(valid_data_size)
with torch.no_grad():
    for spec_blue,spec_red, target,obsid in valid_dataloader:
        spec_blue = spec_blue.to(device)
        spec_red = spec_red.to(device)
        target = target.to(device)
        output = net(spec_blue, spec_red)
        output = torch.max(output, 1)[1]
        obsid = obsid.to(device)
        output_list[valid_step*batch:valid_step*batch+len(target)] = output
        obsid_list[valid_step*batch:valid_step*batch+len(target)] = obsid
        conf_matrix = confusion_matrix(output, target, conf_matrix)
        valid_step = valid_step+1

conf_matrix = np.array(conf_matrix.cpu())
labels = ['OB','AFGK','M']

plt.imshow(conf_matrix, cmap=plt.cm.Blues)

thresh = conf_matrix.max() / 2
for x in range(types_num):
    for y in range(types_num):

        info = int(conf_matrix[y, x])
        plt.text(x, y, info,
                 verticalalignment='center',
                 horizontalalignment='center',
                 color="white" if info > thresh else "black")

ACC = np.zeros((types_num,1))
for i in range(types_num):
    ACC[i] = conf_matrix[i,i]/np.sum(conf_matrix[0:types_num,i])
print('\nACC: {}'.format(ACC))

dataframe = pd.DataFrame({'obsid':obsid_list.int(),'cl1':output_list.int()})

dataframe.set_index('obsid', drop=False, inplace=True)
dataframe = dataframe[~dataframe.index.duplicated(keep="first")]
dataframe.to_csv("cl2.csv",index=False,sep='|')
plt.tight_layout()
plt.yticks(range(types_num), labels)
plt.xticks(range(types_num), labels, rotation=45)
plt.show()
