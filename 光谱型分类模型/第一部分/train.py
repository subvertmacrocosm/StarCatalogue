import os
import time
from torch.utils.data import Dataset, DataLoader
from astropy.io import fits
import numpy as np
import torch
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
    def __init__(self, root, root2):
        self.root = root
        self.root2 = root2
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

            self.set.append({'spec_blue': spec_blue, 'spec_red': spec_red, 'target': target})

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
            self.set.append({'spec_blue': spec_blue, 'spec_red': spec_red, 'target': target})

    def __getitem__(self, index):
        spec_blue = self.set[index]['spec_blue']
        spec_red = self.set[index]['spec_red']
        target = self.set[index]['target']
        return spec_blue, spec_red, target

    def __len__(self):
        return len(self.set)


train_root = 'E:\\DLcode\\dr5data\\st\\StarData\\train\\'
valid_root = 'E:\\DLcode\\dr5data\\st\\StarData\\valid\\'
o_train_root = 'E:\\DLcode\\dr5data\\st\\StarData\\o\\train\\'
o_valid_root = 'E:\\DLcode\\dr5data\\st\\StarData\\o\\valid\\'
batch = 256
learning_rate = 1e-3
epoch = 800

train_dataset = StarDataset(root=train_root,root2=o_train_root)
valid_dataset = StarDataset(root=valid_root,root2=o_valid_root)

train_dataset_size = len(train_dataset)
valid_dataset_size = len(valid_dataset)
print("train_size: {}\nvalid_size: {}".format(train_dataset_size, valid_dataset_size))

train_dataloader = DataLoader(train_dataset, batch_size=batch)
valid_dataloarder = DataLoader(valid_dataset, batch_size=batch)
print("loading data used {}s".format(time.time() - start_time))

net = StarNet()
net = net.to(device)
net.load_state_dict(torch.load('E:\\DLcode\\STARnet\\star_model_602_0.8564395904541016.pth'))


loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)


optimizer = torch.optim.Adamax(net.parameters(), lr=learning_rate)


writer = SummaryWriter("train_logs")
train_step = 0
test_step = 0
total_start_time = time.time()
best_acc = 0

for i in range(epoch):
    print("\n------{}th epoch------".format(i))
    net.train()
    start_time = time.time()
    train_total_loss = 0
    for spec_blue, spec_red, target in train_dataloader:
        spec_blue = spec_blue.to(device)
        spec_red = spec_red.to(device)
        target = target.to(device)

        output = net(spec_blue, spec_red)
        loss = loss_fn(output, target)
        train_total_loss = train_total_loss + loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if train_step % 2000 == 0:
            writer.add_scalar("train_loss", loss.item(), train_step)
        train_step = train_step + 1
    writer.add_scalar("train_total_loss", train_total_loss, test_step)
    net.eval()
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for spec_blue, spec_red , target in valid_dataloarder:
            spec_blue = spec_blue.to(device)
            spec_red = spec_red.to(device)
            target = target.to(device)
            output = net(spec_blue, spec_red)
            loss = loss_fn(output, target)
            total_loss = total_loss + loss.item()
            accuracy = (output.argmax(1) == target).sum()
            total_acc = total_acc + accuracy
    writer.add_scalar("test_loss", total_loss, test_step)
    writer.add_scalar("test_acc", total_acc / valid_dataset_size, test_step)
    end_time = time.time()
    if i == 1:
        torch.save(net, "star_model.pth".format(i))
    if i % 20 == 0:
        torch.save(net.state_dict(), "star_model_{}.pth".format(i))
    if total_acc / valid_dataset_size > best_acc:
        best_acc = total_acc / valid_dataset_size
        torch.save(net.state_dict(), "star_model_{}_{}.pth".format(i,total_acc / valid_dataset_size))
    print("{}th end\nuse_time: {}s  total: {}s".format(test_step, end_time - start_time, end_time - total_start_time))
    print("loss: {}\nACC: {}".format(total_loss, total_acc / valid_dataset_size))
    test_step = test_step + 1

writer.close()