from nets.deeplab.deeplabv3_plus import DeepLab
from nets.unet.unet import Unet
from nets.pspnet.pspnet import PSPNet
import nets.fcn.fcn as fcn
from nets.segnet.segent import SegNet
from nets.new_unet import new_unet
from attention.ema import ema


def create(net_name):
    if net_name == 'unet':
        train_net = new_unet.Unet(num_classes=2).cuda()
    elif net_name == 'ema_unet(g=4)':
        ema.set_groups(4)
        train_net = new_unet.Unet(num_classes=2, vgg_cfg_ind='EMA0').cuda()
    elif net_name == 'ema_unet(g=8)':
        ema.set_groups(8)
        train_net = new_unet.Unet(num_classes=2, vgg_cfg_ind='EMA0').cuda()
    elif net_name == 'ema_unet(g=16)':
        ema.set_groups(16)
        train_net = new_unet.Unet(num_classes=2, vgg_cfg_ind='EMA0').cuda()
    elif net_name == 'eucb_unet':
        train_net = new_unet.Unet(num_classes=2, up_func='eucb').cuda()
    elif net_name == 'eucb_ema_unet(g=4)':
        ema.set_groups(4)
        train_net = new_unet.Unet(num_classes=2, vgg_cfg_ind='EMA0', up_func='eucb').cuda()
    elif net_name == 'eucb_ema_unet(g=8)':
        ema.set_groups(8)
        train_net = new_unet.Unet(num_classes=2, vgg_cfg_ind='EMA0', up_func='eucb').cuda()
    elif net_name == 'eucb_ema_unet(g=16)':
        ema.set_groups(16)
        train_net = new_unet.Unet(num_classes=2, vgg_cfg_ind='EMA0', up_func='eucb').cuda()
    elif net_name == 'mhema_unet':
        train_net = new_unet.Unet(num_classes=2, vgg_cfg_ind='MHEMA20').cuda()
    elif net_name == 'emhseu_unet':
        train_net = new_unet.Unet(num_classes=2, vgg_cfg_ind='MHEMA20', up_func='eucb').cuda()
    elif net_name == 'res_unet':
        train_net = Unet(num_classes=2, backbone='resnet50').cuda()
    elif net_name == 'attention_unet':
        train_net = new_unet.Unet(num_classes=2, is_AG=True).cuda()
    elif net_name == 'deeplabv3+':
        train_net = DeepLab(num_classes=2, backbone='mobilenet', downsample_factor=8, pretrained=False).cuda()
    elif net_name == 'pspnet':
        train_net = PSPNet(num_classes=2, backbone='resnet50', downsample_factor=16, pretrained=False, aux_branch=False).cuda()
    elif net_name == 'fcn':
        train_net = fcn.FCN(backbone='vgg16', num_classes=2).cuda()
    elif net_name == 'segnet':
        train_net = SegNet(2).cuda()

    return train_net