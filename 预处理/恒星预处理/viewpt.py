import torch
import numpy as np
import os
from matplotlib import pyplot as plt
from tqdm.auto import tqdm


index2target = ('O', 'B','A','F','G','K','M')
train_dir = 'E:\\DLcode\\dr5data\\highnors\\train\\A\\'
target_dir = 'E:\\DLcode\\dr5data\\highnors\\jpg\\A\\'
checklist = os.listdir(train_dir)
print('check: {}\n'.format(len(checklist)))
for name in tqdm(checklist, mininterval= 5):
    pt = torch.load(train_dir + name)
    pt[2]
    plt.figure(figsize=(8,4))
    plt.plot(pt[1],pt[0])
    plt.title('{}\nCLASS:{}{}     SNR:{} {} {} {} {}    z: {}     fibermas:{}\nT: {}  Ter: {}'.format(name,index2target[int(pt[2][0]-1)],pt[2][10],int(pt[2][3]),int(pt[2][4]),int(pt[2][5]),int(pt[2][6]),int(pt[2][7]),pt[2][1],pt[2][2],pt[2][8],pt[2][9]))
    # plt.savefig(target_dir + name +'.jpg')
    plt.show()
    plt.close()
print('all done')