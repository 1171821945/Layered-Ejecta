from nets.unet.unet import Unet


def create():
    train_net = Unet(num_classes=2).cuda()
    return train_net