import torch
from torch import nn

class CE_loss(nn.Module):
    def __init__(self):
        super(CE_loss, self).__init__()

    def forward(self, inps, tags):
        n, c, h, w = inps.size()
        temp_inputs = inps.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        temp_target = tags.view(-1)
        loss = nn.CrossEntropyLoss()(temp_inputs, temp_target)
        return loss

class CE_loss(nn.Module):
    def __init__(self, imloss = False):
        super(CE_loss, self).__init__()
        self.imloss = imloss

    def forward(self, inps, tags):
        if not self.imloss:
            n, c, h, w = inps.size()
            temp_inputs = inps.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
            temp_target = tags.view(-1)
            loss = nn.CrossEntropyLoss()(temp_inputs, temp_target)
            return loss
        else:
            num = inps.shape[0]
            loss = 0
            for i in range(0, num):
                c, h, w = inps[i].size()
                inp = inps[i].transpose(0, 1).transpose(1, 2).contiguous().view(-1, c)
                tag = tags[i].view(-1)
                loss += nn.CrossEntropyLoss()(inp, tag) / torch.sum(tag)
            return loss

