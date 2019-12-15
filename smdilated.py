import torch
import torch.nn as nn
import torch.nn.functional as F


class ShareSepConv(nn.Module):
    def __init__(self, kernel_size):
        super(ShareSepConv, self).__init__()
        assert kernel_size % 2 == 1, 'kernel size should be odd'
        self.padding = (kernel_size - 1)//2
        weight_tensor = torch.zeros(1, 1, kernel_size, kernel_size)
        weight_tensor[0, 0, (kernel_size-1)//2, (kernel_size-1)//2] = 1
        self.weight = nn.Parameter(weight_tensor)
        self.kernel_size = kernel_size

    def forward(self, x):
        inc = x.size(1)
        expand_weight = self.weight.expand(inc, 1, self.kernel_size, self.kernel_size).contiguous()
        return F.conv2d(x, expand_weight,
                        None, 1, self.padding, 1, inc)


class SmoothDilated(nn.Module):
    def __init__(self, in_channel, channel_num, dilation=1, group=1):
        super(SmoothDilated, self).__init__()
        self.pre_conv = ShareSepConv(dilation*2-1)
        self.conv = nn.Conv2d(in_channel, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.conv(self.pre_conv(x)))
        return y