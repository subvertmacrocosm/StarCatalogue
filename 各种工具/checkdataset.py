import os
import torch
from tqdm.auto import tqdm
file_dir = 'E:\\DLcode\\dr5data\\st\\StarData\\valid\\'
os.listdir(file_dir)
index2target = ('O', 'B','A','F','G','K','M')
for i in range(7):
    exec("{} = 0".format(index2target[i]))
file_list = os.listdir(file_dir)
for name in tqdm(file_list, mininterval= 5):
    exec("{} = {} + 1".format(index2target[int(torch.load(file_dir + name)[2][0])-1],index2target[int(torch.load(file_dir + name)[2][0])-1]))
print('done\n-----\n')

for i in range(7):
    exec("print({})\n".format(index2target[i]))