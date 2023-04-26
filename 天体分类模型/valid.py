import os
import time

from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from astropy.io import fits
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from model import SpecNet
from sklearn import preprocessing

target2index = {'STAR': 0, 'GALAXY': 1, 'QSO': 2, 'Unknown': 3}
device = torch.device("cpu")
if torch.cuda.is_available() == 1:
    device = torch.device("cuda")

start_time = time.time()


def mms(num, min, max):  # min-max归一化
    return ((num - min) / (max - min))


class Dr5Dataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.set = []

        for root, dirs, files in os.walk(self.root):
            for file in files:
                spec_pth = torch.load(root + '\\' + file)
                target = spec_pth[0][0].int()
                spec_flux = mms(spec_pth[1], spec_pth[1].min(), spec_pth[1].max())
                spec_flux = torch.unsqueeze(spec_flux, 0)  # 增加一个维度表示channel
                self.set.append({'spec': spec_flux, 'target': target})

    def __getitem__(self, index):
        spec_flux = self.set[index]['spec']
        target = self.set[index]['target']
        return spec_flux, target

    def __len__(self):
        return len(self.set)


# 定义参数
valid_root = 'E:\\DLcode\\dr5data\\testvalid\\valid'
batch = 128

# 加载数据集
valid_dataset = Dr5Dataset(root=valid_root)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch)

print("start")
# 创建空的混淆矩阵
types_num = 4
conf_matrix = torch.zeros(types_num, types_num)
conf_matrix = conf_matrix.to(device)

# 载入模型
net = SpecNet()
net = net.to(device)
net.load_state_dict(torch.load('E:\\DLcode\\specnet\\spec_model_1_893.pth'))
print("已载入")

def confusion_matrix(output, targets, conf_matrix):
    output = torch.max(output, 1)[1]
    for out, tar in zip(output, target):
        conf_matrix[out, tar] += 1
    return conf_matrix


net.eval()

with torch.no_grad():
    for spec, target in valid_dataloader:
        spec = spec.to(device)
        target = target.to(device)
        output = net(spec)
        conf_matrix = confusion_matrix(output, target, conf_matrix)

conf_matrix = np.array(conf_matrix.cpu())
labels = ['STAR', 'GALAXY', 'QSO', 'Unknown']  # 每种类别的标签

# 显示数据
plt.imshow(conf_matrix, cmap=plt.cm.Blues)

# 在图中标注数量/概率信息
thresh = conf_matrix.max() / 2  # 数值颜色阈值，如果数值超过这个，就颜色加深。
for x in range(types_num):
    for y in range(types_num):
        # 注意这里的matrix[y, x]不是matrix[x, y]
        info = int(conf_matrix[y, x])
        plt.text(x, y, info,
                 verticalalignment='center',
                 horizontalalignment='center',
                 color="white" if info > thresh else "black")

plt.tight_layout()
plt.yticks(range(types_num), labels)
plt.xticks(range(types_num), labels, rotation=45)  # X轴字体倾斜45°
plt.show()
