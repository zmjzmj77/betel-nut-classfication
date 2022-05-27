import torch
import torch.nn as nn
import torch.functional as F
import torchsummary


class BetelNet(nn.Module):
    def __init__(self, in_channels, BN=True, f_flag='swish'):
        super(BetelNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 14, kernel_size=(1, 1), stride=1)
        self.bn1 = nn.BatchNorm2d(14)

        self.conv2 = nn.Conv2d(14, 20, kernel_size=(1, 1), stride=1)
        self.bn2 = nn.BatchNorm2d(20)

        self.conv3 = nn.Conv2d(20, 25, kernel_size=(1, 1), stride=1)
        self.bn3 = nn.BatchNorm2d(25)

        self.conv4 = nn.Conv2d(25, 40, kernel_size=(1, 1), stride=1)
        self.bn4 = nn.BatchNorm2d(40)

        self.conv5 = nn.Conv2d(40, 5, kernel_size=(1, 1), stride=1)

        self.fc1 = nn.Linear(40, 5, bias=True)
        # self.fc2 = nn.Linear(20, 5, bias=True)

        self.dropout = nn.Dropout(p=0.2)

        self.relu = torch.nn.ReLU()
        self.elu = torch.nn.ELU()
        self.lrelu = torch.nn.LeakyReLU()
        self.prelu = torch.nn.PReLU()

        self.bn_flag = BN
        self.f_flag = f_flag

    def Swish(self, x, beta):
        return x * torch.sigmoid(beta * x)

    def select_f(self, x):
        if self.f_flag == 'swish':
            return self.Swish(x, 1.0)
        elif self.f_flag == 'prelu':
            return self.prelu(x)
        else:
            return self.relu(x)

    def l1_attention(self, x):
        mask = torch.sigmoid(x)
        B, _, H, W = mask.shape
        norm = torch.norm(mask, p=1, dim=(1,2,3))
        norm = norm.reshape(B, 1, 1, 1)
        mask = torch.div(mask * H * W, norm)
        x = torch.mul(x, mask)
        return x

    def se_attention(self, x):
        mask = torch.sigmoid(x)
        x = torch.mul(x, mask)
        return x

    def forward(self, x):
        x = self.conv1(x)
        if self.bn_flag:
            x = self.bn1(x)
        x = self.select_f(x)

        x = self.conv2(x)
        if self.bn_flag:
            x = self.bn2(x)
        x = self.select_f(x)

        # x = self.se_attention(x)

        x = self.conv3(x)
        if self.bn_flag:
            x = self.bn3(x)
        x = self.select_f(x)

        x = self.conv4(x)
        if self.bn_flag:
            x = self.bn4(x)
        x = self.select_f(x)

        # x = self.se_attention(x)

        x = torch.flatten(x, 1)
        # x = self.dropout(x)
        x = self.fc1(x)

        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    model = BetelNet(8).cuda()
    torchsummary.summary(model, (8,1,1))
    print('parameters_count:', count_parameters(model))

