import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=(5, 5), stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d(kernel_size=(2,2))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(kernel_size=(2, 2))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.AvgPool2d(kernel_size=(2,2))
        self.relu = nn.ReLU()

        self.conv11 = nn.Conv2d(32, 64, kernel_size=(1, 1))
        self.conv22 = nn.Conv2d(64, 128, kernel_size=(1, 1))

        # self.basenet = nn.Sequential(
        #     nn.Conv2d(3, 6, kernel_size=(5, 5), stride=1, padding=2),
        #     nn.BatchNorm2d(6),
        #     nn.ReLU(),
        #     nn.AvgPool2d(kernel_size=(2,2)),
        #     nn.Conv2d(6, 12, kernel_size=(3, 3), stride=1, padding=1),
        #     nn.BatchNorm2d(12),
        #     nn.ReLU(),
        #     nn.AvgPool2d(kernel_size=(2, 2)),
        #     nn.Conv2d(12, 24, kernel_size=(3, 3), stride=1, padding=1),
        #     nn.BatchNorm2d(24),
        #     nn.ReLU(),
        #     nn.AvgPool2d(kernel_size=(2,2))
        # )
        self.classfier = nn.Sequential(
            nn.Linear(128 ** 2, 5),
            nn.ReLU()
        )

    def forward(self, x):
        # x = self.basenet(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)
        x = self.conv11(x)
        x += out
        x = self.pool2(x)
        out = self.conv3(x)
        out = self.bn3(out)
        out = self.relu(out)
        x = self.conv22(x)
        x += out
        x = self.pool3(x)

        batch_size = x.size(0)
        x = x.view(batch_size, 128, -1)
        x = (torch.bmm(x,torch.transpose(x, 1, 2)) / 28 ** 2).view(batch_size, -1)
        x = torch.nn.functional.normalize(torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10))
        x = self.classfier(x)
        return x


