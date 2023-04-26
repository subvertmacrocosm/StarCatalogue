import torch
import torch.nn.functional as F

class StarNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model_blue = torch.nn.Sequential(
            torch.nn.Conv1d(1, 16, 7, 2, padding=3),
            torch.nn.MaxPool1d(2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 64, 3, padding=1),
            torch.nn.MaxPool1d(2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Flatten(),
        )
        self.model_red = torch.nn.Sequential(
            torch.nn.Conv1d(1, 16, 7, 2, padding=3),
            torch.nn.MaxPool1d(2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 64, 3, padding=1),
            torch.nn.MaxPool1d(2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Flatten(),
        )
        self.linear_net = torch.nn.Sequential(
            torch.nn.Linear(9536, 1024),
            torch.nn.Linear(1024, 256),
            torch.nn.Linear(256, 64),
            torch.nn.Linear(64, 7),
        )

    def forward(self, input_blue, input_red):
        blue = self.model_blue(input_blue)
        red = self.model_red(input_red)
        x = torch.cat((blue,red),1)
        x = self.linear_net(x)
        x = F.softmax(x,dim=1)
        return x


if __name__ == '__main__':
    detect = StarNet()
    input_blue = torch.ones((1, 1, 1900))
    input_red = torch.ones((1, 1, 500))
    output = detect(input_blue,input_red)
    print(output.shape)
    print(output)
