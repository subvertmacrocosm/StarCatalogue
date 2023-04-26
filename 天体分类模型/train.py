import os
import time
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
            for file in np.random.choice(files,10):
                spec1 = fits.open(root + '\\' + file)
                target = target2index[spec1[0].header['CLASS']]
                spec = spec1[0].data[0]
                spec = spec.byteswap().newbyteorder()
                if len(spec) < 3909:
                    spec = np.pad(spec, (0, 3909 - len(spec)), 'constant', constant_values=(0, 0))  # 补0
                spec = mms(spec, spec[spec.nonzero()].min(), np.max(spec))
                spec = np.maximum(spec, 0)
                spec = spec[np.newaxis, :]  # 增加一个维度表示channel
                spec = torch.tensor(spec)
                target = torch.tensor(target)
                self.set.append({'spec': spec, 'target': target})

    def __getitem__(self, index):
        spec = self.set[index]['spec']
        target = self.set[index]['target']
        return spec, target

    def __len__(self):
        return len(self.set)


# 定义参数
train_root = 'E:\\DLcode\\dr5data\\dr5_52453\\train'
test_root = 'E:\\DLcode\\dr5data\\dr5_52453\\test'
batch = 128
learning_rate = 1e-3  # 5e-6太小了
epoch = 5000

# 加载数据集
train_dataset = Dr5Dataset(root=train_root)
test_dataset = Dr5Dataset(root=test_root)

train_dataset_size = len(train_dataset)
test_dataset_size = len(test_dataset)
print("训练集大小:{}\n测试集大小：{}".format(train_dataset_size, test_dataset_size))

train_dataloader = DataLoader(train_dataset, batch_size=batch)
test_dataloarder = DataLoader(test_dataset, batch_size=batch)
print("加载数据耗时{}秒".format(time.time()-start_time))

# 初始化网络
net = SpecNet()
net = net.to(device)
# net.load_state_dict(torch.load('E:\\DLcode\\specnet\\spec_model_1_893.pth'))


# 损失函数
loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
optimizer = torch.optim.Adamax(net.parameters(), lr=learning_rate)

# 记录器
writer = SummaryWriter("train_logs")
train_step = 0
test_step = 0
total_start_time = time.time()
for i in range(epoch):
    print("\n------第{}轮训练开始------".format(i))
    net.train()
    start_time = time.time()
    for spec, target in train_dataloader:
        spec = spec.to(device)
        target = target.to(device)

        # 前向传播
        output = net(spec)
        loss = loss_fn(output, target)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if train_step % 2000 == 0:
            writer.add_scalar("train_loss", loss.item(), train_step)
        train_step = train_step + 1

    # 测试
    net.eval()
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for spec, target in test_dataloarder:
            spec = spec.to(device)
            target = target.to(device)
            output = net(spec)
            loss = loss_fn(output, target)
            total_loss = total_loss + loss.item()
            accuracy = (output.argmax(1) == target).sum()
            total_acc = total_acc + accuracy
    writer.add_scalar("test_loss", total_loss, test_step)
    writer.add_scalar("test_acc", total_acc / test_dataset_size, test_step)
    end_time = time.time()
    # 保存模型
    torch.save(net.state_dict(), "spec_model_1_{}.pth".format(i))
    print("第{}轮训练结束\n耗时{}秒，总耗时{}秒".format(test_step, end_time - start_time, end_time - total_start_time))
    print("loss: {}\nACC: {}".format(total_loss, total_acc / test_dataset_size))
    test_step = test_step + 1

writer.close()
