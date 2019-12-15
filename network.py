import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from smdilated import SmoothDilated
from ResCBAM import ResCBAMGroup
from utils import *

class RDARENet(nn.Module):
    def __init__(self, recurrent_iter=6, residual_iter=6, use_GPU=True):
        super(RDARENet, self).__init__()

        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.smdilated1 = SmoothDilated(6, 32, dilation=1)
        self.smdilated2 = SmoothDilated(6, 32, dilation=2)
        self.smdilated3 = SmoothDilated(6, 32, dilation=3)

        self.smcat = nn.Conv2d(32 * 3, 32, 3, 1, 1)

        # LSTM
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )

        self.rescbamgroup = ResCBAMGroup(32, 3, 16, residual_iter)
        self.conv = nn.Conv2d(32, 3, 3, 1, 1)
        self.gate_fuse = nn.Conv2d(3 * self.iteration, self.iteration, 1, 1, 0, bias=True)

    def forward(self, input):

        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))
        out = Variable(torch.zeros(batch_size, 3, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()
            out = out.cuda()

        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            y1 = self.smdilated1(x)
            y2 = self.smdilated2(x)
            y3 = self.smdilated3(x)

            concat = self.smcat(torch.cat((y1, y2, y3), dim=1))
            y = F.relu(concat)

            x = torch.cat((y, h), dim=1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            x = self.rescbamgroup(x)
            x = self.conv(x)

            x = x + input
            x_list.append(x)

        x_sum = self.gate_fuse(torch.cat(x_list, dim=1))
        for idx, x_idx in enumerate(x_list):
            out += x_idx * x_sum[:, [idx], :, :]

        return out, x_list



if __name__ == '__main__':

    model = RDARENet()
    print_network(model)




