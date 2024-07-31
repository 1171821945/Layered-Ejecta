import torch
from torch import nn



def getIOU_(inp, tag):
    inp = inp.transpose(0, 1).transpose(1, 2).contiguous()
    inp = torch.argmax(torch.softmax(inp, -1), -1)
    I = torch.sum(inp * tag)
    U = torch.sum(inp) + torch.sum(tag) - I
    return I / U


def getCE(inp, tag, is_imp_loss):
    c, h, w = inp.size()
    inp = inp.transpose(0, 1).transpose(1, 2).contiguous().view(-1, c)
    tag = tag.view(-1)
    if is_imp_loss:
        return nn.CrossEntropyLoss()(inp, tag) / torch.sum(tag)
    else:
        return nn.CrossEntropyLoss()(inp, tag)


def getCEs(inps, tags, is_imp_loss):
    num = inps.shape[0]
    loss = 0
    for i in range(0, num):
        loss = loss+getCE(inps[i], tags[i], is_imp_loss)
    return loss/num


def get_IOU_loss(inps, tags):
    loss = 0
    num = inps.shape[0]
    for i in range(0, num):
        loss = loss + getIOU_(inps[i], tags[i])
    return loss/num


class EC_loss(nn.Module):

    def __init__(self, is_imp_loss=True):
        super(EC_loss, self).__init__()
        self.is_imp_loss = is_imp_loss

    def forward(self, inps, tags):
        return  getCEs(inps, tags, self.is_imp_loss)




class IOU_loss(nn.Module):

    def __init__(self):
        super(IOU_loss, self).__init__()

    def getIOU_(self, inp, tag):
        inp = inp.transpose(0, 1).transpose(1, 2).contiguous()
        inp = torch.argmax(torch.softmax(inp, -1), -1)
        # I = torch.sum(inp * tag)
        # U = torch.sum(inp) + torch.sum(tag) - I
        return torch.sum(inp)

    def get_IOU_loss(self, inps, tags):
        loss = 0
        num = inps.shape[0]
        for i in range(0, num):
            loss = loss + self.getIOU_(inps[i], tags[i])
        return loss

    def forward(self, inps, tags):
        return 1-self.get_IOU_loss(inps, tags)


