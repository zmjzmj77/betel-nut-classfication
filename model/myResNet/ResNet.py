import os

import torch.nn as nn
import torch
from torchvision.models.utils import load_state_dict_from_url
from model.registry import register_model
import torch.nn.functional as F
from cfg.cfg import parameters as cfg

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)




class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, attention = False):
        super(BasicBlock, self).__init__()
        self.attention = attention
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        # self.conv7 = nn.Conv2d(2, 1, (7,7), padding=3, bias=False)  # 输入两个通道，一个是maxpool 一个是avgpool的
        # self.sigmoid = nn.Sigmoid()

    # def spa_attention(self, x):
    #     avg_out = torch.mean(x, dim=1, keepdim=True)
    #     max_out, _ = torch.max(x, dim=1, keepdim=True)
    #     x = torch.cat([avg_out, max_out], dim=1)
    #     x = self.conv7(x)  # 对池化完的数据cat 然后进行卷积
    #     x = self.sigmoid(x)
    #     return x

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        # if self.attention is True:
        #     old = out
        #     out = self.spa_attention(out)
        #     out = out * old

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, attention = None, num_classes=5,  zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.block = block
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], attention=attention)
        self.new_layer2 = self._make_layer(block, 96, layers[1], attention=attention, stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.new_layer3 = self._make_layer(block, 128, layers[2], attention=attention, stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.new_layer4 = self._make_layer(block, 160, layers[3], attention=attention, stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.layer5_1 = nn.Sequential(
            nn.Conv2d(160, 192, kernel_size=(3, 3), padding=1),
            norm_layer(192),
        )
        self.layer5_2 = nn.Sequential(
            nn.Conv2d(160, 192, kernel_size=(1, 1)),
            norm_layer(192),
        )

        self.conv7 = nn.Conv2d(2, 1, (3, 3), padding=1, bias=False)  # 输入两个通道，一个是maxpool 一个是avgpool的
        self.sigmoid = nn.Sigmoid()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.new_fc = nn.Linear(512 * block.expansion, num_classes)
        self.new_fc = nn.Linear(192** 2, num_classes)

        self.dropout = nn.Dropout(p=0.3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks,attention=False, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, attention=attention))

        return nn.Sequential(*layers)

    def spa_attention(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv7(x)  # 对池化完的数据cat 然后进行卷积
        x = self.sigmoid(x)
        return x

    def calculate_gamma(self, x, block_size, p):
        invalid = (1 - p) / (block_size ** 2)
        valid = (x.shape[-1] ** 2) / ((x.shape[-1] - block_size + 1) ** 2)
        return invalid * valid

    def dropblock(self, x, block_size, p) :
        gamma = self.calculate_gamma(x, block_size, p)
        mask = torch.bernoulli(torch.ones_like(x) * gamma)
        mask_block = 1 - F.max_pool2d(
            mask,
            kernel_size=(block_size, block_size),
            stride=(1, 1),
            padding=(block_size // 2,block_size // 2),
        )
        x = mask_block * x * (mask_block.numel() / mask_block.sum())
        return x

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.new_layer2(x)
        x = self.new_layer3(x)
        # x = self.dropblock(x, block_size=3, p=0.3)
        x = self.new_layer4(x)

        x_1 = x
        x_2 = x
        x_1 = self.layer5_1(x_1)
        x_1 = self.relu(x_1)

        x_2 = self.layer5_2(x_2)
        x_2_old = x_2
        x_2 = self.spa_attention(x_2)
        x_2 = x_2_old * x_2
        # if cfg['visiual']:
        #     feature_map = x_2
        #     path = '/home/zhaomengjun/2021_binglang_paper/paper_code/logs/resnest18/feature_map'
        #     n = len(os.listdir(path))
        #     import matplotlib.pyplot as plt
        #     # plt.figure(figsize=(224,224))
        #     upsample = torch.nn.UpsamplingBilinear2d(size=(224,224))  # 这里进行调整大小
        #     feature_map = upsample(feature_map)
        #     feature_map = feature_map.view(feature_map.shape[1], feature_map.shape[2], feature_map.shape[3])
        #     # print(type(feature_map[0]), feature_map[0].shape)
        #     plt.imshow(feature_map[0].cpu().detach().numpy(), cmap='gray')
        #     plt.savefig(os.path.join(path,'{}.png'.format(n+1)))
        x_2 = self.relu(x_2)


        batch_size, c, h = x_1.size(0), x_1.size(1), x_1.size(2)
        x_1 = x_1.view(batch_size, c, -1)
        x_2 = x_2.view(batch_size, c, -1)

        # x = self.avgpool(x)

        x = (torch.bmm(x_1, torch.transpose(x_2, 1, 2)) / h ** 2).view(batch_size, -1)
        x = torch.nn.functional.normalize(torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10))
        x = self.dropout(x)
        # x = torch.flatten(x, 1)
        x = self.new_fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


@register_model
def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2,1,1,1], pretrained, progress, False,
                   **kwargs)


@register_model
def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


@register_model
def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


@register_model
def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


@register_model
def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def _resnet(arch, block, layers, pretrained, progress, attention, **kwargs):
    model = ResNet(block, layers,attention, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    num_classes = 4
    x = torch.rand([4, 3, 224, 224])
    model = resnet152()
    # print(model)
    out = model(x)
