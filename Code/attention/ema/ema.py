import torch
from torch import nn
class EMA(nn.Module):
    def __init__(self, channels, factor=4):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class MHEMA(nn.Module):
    def __init__(self, channels):
        super(MHEMA, self).__init__()
        self.EMA0 = EMA(channels).cuda()
        self.EMA1 = EMA(channels).cuda()
        self.EMA2 = EMA(channels).cuda()
        self.prj = torch.nn.Conv2d(channels*3, channels, kernel_size=3, padding=1)
    def forward(self, x):
        x0 = self.EMA0(x)
        x1 = self.EMA1(x)
        x2 = self.EMA2(x)
        x = torch.concatenate([x0, x1, x2], 1)
        x = self.prj(x)
        return x

class MHEMA2(nn.Module):
    def __init__(self, channels):
        super(MHEMA2, self).__init__()
        self.EMA0 = EMA(channels, factor=4).cuda()
        self.EMA1 = EMA(channels, factor=8).cuda()
        self.EMA2 = EMA(channels, factor=16).cuda()
        self.prj = torch.nn.Conv2d(channels*3, channels, kernel_size=1, padding=0)
    def forward(self, x):
        x0 = self.EMA0(x)
        x1 = self.EMA1(x)
        x2 = self.EMA2(x)
        x = torch.concatenate([x0, x1, x2], 1)
        x = self.prj(x)
        return x


class MHEMA_sum(nn.Module):
    def __init__(self, channels):
        super(MHEMA_sum, self).__init__()
        self.EMA0 = EMA(channels).cuda()
        self.EMA1 = EMA(channels).cuda()
        self.EMA2 = EMA(channels).cuda()
    def forward(self, x):
        x0 = self.EMA0(x)
        x1 = self.EMA1(x)
        x2 = self.EMA2(x)
        x = x0 + x1 + x2
        return x


class MHEMA_sum2(nn.Module):
    def __init__(self, channels):
        super(MHEMA_sum2, self).__init__()
        self.EMA0 = EMA(channels, factor=4).cuda()
        self.EMA1 = EMA(channels, factor=8).cuda()
        self.EMA2 = EMA(channels, factor=16).cuda()
    def forward(self, x):
        x0 = self.EMA0(x)
        x1 = self.EMA1(x)
        x2 = self.EMA2(x)
        x = x0 + x1 + x2
        return x


# 输入 B C H W,  输出 B C H W
if __name__ == '__main__':
    block = MHEMA(64).cuda()
    input = torch.rand(1, 64, 64, 64).cuda()
    output = block(input)
    print(input.size(), output.size())
