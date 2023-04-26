import torch


class SpecNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv1d(1, 8, 7, 2, padding=3),
            torch.nn.MaxPool1d(2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(8, 16, 3, padding=1),
            torch.nn.MaxPool1d(2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(7808, 256),
            torch.nn.Linear(256, 64),
            torch.nn.Linear(64, 3),
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    detect = SpecNet()
    input = torch.ones((1, 1, 3909))
    output = detect(input)
    print(output.shape)
    print(output)
