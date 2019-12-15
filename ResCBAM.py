import torch.nn as nn
import torch.nn.functional as F

from attention import ChannelAttention, SpatialAttention


# Residual Convolution Block Attention Module
class ResCBAM(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias=True):
        super(ResCBAM, self).__init__()

        modules_head = []
        for i in range(2):
            modules_head.append(nn.Conv2d(in_channels=n_feat,out_channels=n_feat, kernel_size=kernel_size, padding=1, bias=bias))
            if i == 0: modules_head.append(nn.ReLU(True))

        self.head = nn.Sequential(*modules_head)
        self.ca = ChannelAttention(n_feat, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):

        res = self.head(x)
        ca_out = self.ca(res)
        sa_out = self.sa(ca_out)
        res = sa_out + x
        res = F.relu(res)
        return res


# A Group Of ResCBAM
class ResCBAMGroup(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, n_resblocks):
        super(ResCBAMGroup, self).__init__()
        modules_body =[
            ResCBAM(n_feat, kernel_size, reduction, bias=True)
            for _ in range(n_resblocks)]
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        res = F.relu(res)
        return res

