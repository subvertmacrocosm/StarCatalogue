import os
import time

from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from astropy.io import fits
import numpy as np
import torch
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from model import StarNet
from sklearn import preprocessing
from tqdm.auto import tqdm

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

        for name in tqdm(os.listdir(self.root), mininterval=3):
            spec1 = torch.load(root + name)
            target = spec1[2][8]
            spec_blue = spec1[0][(spec1[1] > 3900) & (spec1[1] <= 5900)].float()
            spec_red = spec1[0][(spec1[1] >= 7950) & (spec1[1] < 8950)].float()
            spec_blue = torch.nan_to_num(spec_blue,nan=1.0, posinf=1.0, neginf=1.0)
            spec_red = torch.nan_to_num(spec_red, nan=1.0, posinf=1.0, neginf=1.0)
            if len(spec_blue)<2000:
                spec_blue = torch.cat((torch.ones(2000-len(spec_blue)),spec_blue), dim=0)
            if len(spec_red)<1000:
                spec_red = torch.cat((torch.ones(1000 - len(spec_red)), spec_red), dim=0)
            spec_blue = mms(spec_blue, spec_blue.min(), spec_blue.max()).unsqueeze(0)
            spec_red = mms(spec_red, spec_red.min(), spec_red.max()).unsqueeze(0)
            spec_red = torch.nan_to_num(spec_red, nan=1.0, posinf=1.0, neginf=1.0)
            obsid = spec1[2][11]
            class_target  = spec1[2][0].type(torch.int64) - 3
            self.set.append({'spec_blue': spec_blue, 'spec_red': spec_red, 'target': target, 'obsid': obsid, 'class_target': class_target})

    def __getitem__(self, index):
        spec_blue = self.set[index]['spec_blue']
        spec_red = self.set[index]['spec_red']
        target = self.set[index]['target']
        obsid = self.set[index]['obsid']
        class_target = self.set[index]['class_target']
        return spec_blue, spec_red, target , obsid, class_target

    def __len__(self):
        return len(self.set)



valid_root = 'E:\\DLcode\\dr5data\\temdataset\\新建文件夹\\'



batch = 256

types_num = 4
conf_matrix = torch.zeros(types_num, types_num)

def confusion_matrix(output, target, conf_matrix):
    for out, tar in zip(output, target):
        # if (out == 0) & (output[i][0] < 0.9999): #
        #     out = 3
        out = out.cpu()
        tar = tar.cpu()
        conf_matrix[int(out), int(tar)] += 1
    return conf_matrix

conf_matrix = np.array(conf_matrix.cpu())


valid_dataset = StarDataset(root=valid_root)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch)
valid_dataset_size = len(valid_dataset)
valid_dataset_size = len(valid_dataset)
print("valid_size: {}".format( valid_dataset_size))
print("start")


dicname = 'nb_star_dic_722_23.90445658125007.pth'
net = torch.load('E:\\DLcode\\temPredictt\\star_model.pth')
net = net.to(device)
net.load_state_dict(torch.load('E:\\DLcode\\temPredictt\\'+dicname))

target_list = torch.zeros(1,valid_dataset_size)
output_list = torch.zeros(1,valid_dataset_size)
obsid_list = torch.zeros(1,valid_dataset_size)

otA = torch.zeros(0).to(device)
otF = torch.zeros(0).to(device)
otG = torch.zeros(0).to(device)
otK = torch.zeros(0).to(device)
taA = torch.zeros(0).to(device)
taF = torch.zeros(0).to(device)
taG = torch.zeros(0).to(device)
taK = torch.zeros(0).to(device)
obs = torch.zeros(0).to(device).int()
net.eval()
valid_step = 0
with torch.no_grad():
    for spec_blue,spec_red, target, obsid ,class_target in valid_dataloader:
        spec_blue = spec_blue.to(device)
        spec_red = spec_red.to(device)
        target = target.to(device)
        obsid = obsid.to(device)
        target_list[0,valid_step*batch:valid_step*batch+len(target)] = target
        obsid_list[0,valid_step*batch:valid_step*batch+len(target)] = obsid
        output = net(spec_blue, spec_red)

        class_output = torch.where(output>7500, 0, output)
        ot = torch.squeeze(output,dim=1)
        class_target = class_target.to(device)
        otA  = torch.cat((otA,torch.index_select(output,dim=0,index=torch.nonzero(class_target==0).squeeze())))
        taA = torch.cat((taA,torch.index_select(target,dim=0,index=torch.nonzero(class_target==0).squeeze()).to(torch.float32)))
        otF = torch.cat((otF,torch.index_select(output, dim=0, index=torch.nonzero(class_target==1).squeeze())))
        taF = torch.cat((taF,torch.index_select(target,dim=0,index=torch.nonzero(class_target==1).squeeze()).to(torch.float32)))
        otG = torch.cat((otG,torch.index_select(output, dim=0, index=torch.nonzero(class_target==2).squeeze())))
        taG = torch.cat((taG,torch.index_select(target,dim=0,index=torch.nonzero(class_target==2).squeeze()).to(torch.float32)))
        otK = torch.cat((otK,torch.index_select(output, dim=0, index=torch.nonzero(class_target==3).squeeze())))
        taK = torch.cat((taK,torch.index_select(target,dim=0,index=torch.nonzero(class_target==3).squeeze()).to(torch.float32)))
        # otA  = torch.cat((otA,torch.index_select(output,dim=0,index=torch.nonzero(ot>=7500).squeeze())))
        # taA = torch.cat((taA,torch.index_select(target,dim=0,index=torch.nonzero(ot>=7500).squeeze()).to(torch.float32)))
        # otF = torch.cat((otF,torch.index_select(output, dim=0, index=torch.nonzero((ot >= 6000) & (ot < 7500)).squeeze())))
        # taF = torch.cat((taF,torch.index_select(target, dim=0, index=torch.nonzero((ot >= 6000) & (ot < 7500)).squeeze()).to(torch.float32)))
        # otG = torch.cat((otG,torch.index_select(output, dim=0, index=torch.nonzero((ot >= 4900) & (ot < 6000)).squeeze())))
        # taG = torch.cat((taG,torch.index_select(target, dim=0, index=torch.nonzero((ot >= 4900) & (ot < 6000)).squeeze()).to(torch.float32)))
        # otK = torch.cat((otK,torch.index_select(output, dim=0, index=torch.nonzero((ot > 10) & (ot < 4900)).squeeze())))
        # taK = torch.cat((taK,torch.index_select(target, dim=0, index=torch.nonzero((ot >= 10) & (ot < 4900)).squeeze()).to(torch.float32)))
        class_output = torch.where(class_output > 6000, 1, class_output)
        class_output = torch.where(class_output > 4900, 2, class_output)
        class_output = torch.where(class_output > 10, 3, class_output)
        class_output.cpu()
        class_target.cpu()
        conf_matrix = confusion_matrix(class_output, class_target, conf_matrix)

        output_list[0,valid_step * batch:valid_step * batch + len(target)] =  ot

        torch.nonzero(torch.abs(ot-target)>100)
        aa= torch.nonzero(torch.abs(ot-target)>100,as_tuple=False).squeeze()
        obserr = torch.index_select(obsid, 0, index = aa).int()
        obs = torch.cat([obs,obserr], dim=0)
        valid_step = valid_step +1


labels = ['A','F','G','K']
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

plt.tight_layout()
plt.yticks(range(types_num), labels)
plt.xticks(range(types_num), labels, rotation=45)
plt.show()

plt.plot(target_list[0],output_list[0],'ro')
plt.title(dicname)
plt.show()

plt.plot(target_list[0],target_list[0]-output_list[0],'ro')
plt.show()
aaa=target_list[0]-output_list[0]
aaa = aaa.to(device)
mm1 = aaa.mean()
st3 = 3*torch.std(aaa)
idx = torch.nonzero(abs(aaa-mm1)<st3).squeeze()
bbb = torch.index_select(aaa,dim=0,index = idx)

import pandas as pd

data = pd.Series(bbb.cpu())
plt.hist(data, density=True, edgecolor='w', label='histogram')
data.plot(kind='kde', label='density')


plt.legend()

plt.show()

dataframe = pd.DataFrame({'obsid':obsid_list[0].int(),'T':output_list[0]})
dataframe.set_index('obsid', drop=False, inplace=True)
dataframe = dataframe[~dataframe.index.duplicated(keep="first")]
dataframe.to_csv("st3.csv",sep='|')

plt.plot(taA.cpu(),otA.cpu(),'r+')
plt.plot(taF.cpu(),otF.cpu(),'b+')
plt.plot(taG.cpu(),otG.cpu(),'g+')
plt.plot(taK.cpu(),otK.cpu(),'y+')
plt.show()
plt.savefig('E:\\DLcode\\temPredictt\\jpg_{}.jpg'.format(dicname))
torch.save(obs,'E:\\DLcode\\temPredictt\\err100_{}.pt'.format(dicname))

print(len(obs))
