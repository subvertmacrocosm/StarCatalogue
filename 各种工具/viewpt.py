import torch
import numpy as np
import os
from matplotlib import pyplot as plt
from tqdm.auto import tqdm


index2target = ('O', 'B','A','F','G','K','M')
train_dir = 'E:\\DLcode\\dr5data\\st\\noredshift\\valid\\'
target_dir = 'E:\\DLcode\\dr5data\\st\\noredshift\\jpg\\'
checklist = os.listdir(train_dir)
print('check: {}\n'.format(len(checklist)))
for name in tqdm(checklist, mininterval= 5):
    pt = torch.load(train_dir + name)
    pt[2]
    plt.figure(figsize=(8,4))
    plt.plot(pt[0],pt[1])
    plt.title('CLASS:{}     SNR:{} {} {} {} {}    z: {}     fibermas:{}\n{}'.format(index2target[int(pt[2][0]-1)],pt[2][3],pt[2][4],pt[2][5],pt[2][6],pt[2][7],pt[2][1],pt[2][2],name))
    plt.savefig(target_dir + name +'.jpg')
    plt.close()
print('all done')