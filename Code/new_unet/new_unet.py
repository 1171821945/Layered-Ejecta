import torch
import torch.nn as nn
from attention.ema.ema import EMA, MHEMA, MHEMA_sum
from nets.unet.resnet import resnet50
from nets.new_unet.nu_vgg import VGG16
from modules.CGA import CGA
from attention.aa.aa import AgentAttention
from modules.EUCB import  EUCB
from attention.DA_Block import DA_Block
from modules.FEM import FEM

class AttentionBlock(nn.Module):
    def __init__(self, skip_x_channels, x_channels, int_channels, is_batch = False):
        super(AttentionBlock, self).__init__()
        if is_batch:
            self.W_skip_x = nn.Sequential(nn.Conv2d(skip_x_channels, int_channels, kernel_size=1),
                                          nn.BatchNorm2d(int_channels))
            self.Wx = nn.Sequential(nn.Conv2d(x_channels, int_channels, kernel_size=1),
                                    nn.BatchNorm2d(int_channels))
            self.psi = nn.Sequential(nn.Conv2d(int_channels, 1, kernel_size=1),
                                     nn.BatchNorm2d(1),
                                     nn.Sigmoid())
        else:
            self.W_skip_x = nn.Conv2d(skip_x_channels, int_channels, kernel_size=1)
            self.Wx = nn.Conv2d(x_channels, int_channels, kernel_size=1)
            self.psi = nn.Sequential(nn.Conv2d(int_channels, 1, kernel_size=1),
                                     nn.Sigmoid())


    def forward(self, skip, x):
        # apply the W_skip_x to the skip connection
        W_skip_x1 = self.W_skip_x(skip)
        # after applying Wx to the input, upsample to the size of the skip connection
        Wx1 = nn.functional.interpolate(self.Wx(x), W_skip_x1.shape[2:], mode='bilinear', align_corners=False)
        out = self.psi(nn.ReLU()(W_skip_x1 + Wx1))
        return out * skip


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_batch = False, is_AG=False, fu_func = 'cat', up_func = 'bl', skip_attention = 'none'):
        super(unetUp, self).__init__()
        self.is_AG= is_AG
        self.fu_func = fu_func
        self.skip_attention = skip_attention
        # 连接方式
        if fu_func == 'cat' or fu_func == 'aa':
            conv1_layers = [nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)]
        elif fu_func == 'cga':
            conv1_layers = [nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)]
        else:
            raise Exception


        if self.skip_attention != 'none':
            if self.skip_attention == 'ema':
                self.skip_attention = EMA(out_size)
            elif self.skip_attention == 'da':
                self.skip_attention = DA_Block(out_size)
            elif self.skip_attention == 'fem':
                self.skip_attention = FEM(out_size, out_size)
            else :
                raise Exception

        conv2_layers = [nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)]
        if is_batch:
            conv1_layers += [nn.BatchNorm2d(out_size)]
            conv2_layers += [nn.BatchNorm2d(out_size)]
        conv1_layers += [nn.ReLU(inplace=True)]
        conv2_layers += [nn.ReLU(inplace=True)]
        self.conv1 = nn.Sequential(*conv1_layers)
        self.conv2 = nn.Sequential(*conv2_layers)
        if up_func == 'bl':
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        elif up_func == 'eucb':
            self.up = EUCB(in_size - out_size, in_size - out_size)

        if is_AG:
            self.ag = AttentionBlock(out_size, in_size - out_size, int((in_size - out_size) / 2))
        if fu_func == 'cga':
            self.CGA = CGA(out_size, in_size - out_size)
        if fu_func == 'aa':
            self.AA = AgentAttention(in_size * ((4096 // out_size) ** 2), 16 * 16)

    def forward(self, skip, x):

        if self.skip_attention != 'none':
            skip = self.skip_attention(skip)

        if self.is_AG:
            skip = self.ag(skip, x)

        if self.fu_func == 'cat':
            outputs = torch.cat([skip, self.up(x)], 1)
        elif self.fu_func == 'cga':
            outputs = self.CGA(skip, self.up(x))
        elif self.fu_func == 'aa':
            outputs = torch.cat([skip, self.up(x)], 1)
            outputs = self.AA(outputs)

        else :
            raise Exception
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)

        return outputs



class Unet(nn.Module):
    # isbatch: if bn?
    # is_body_batch net(VGG)
    def __init__(self, num_classes = 21, pretrained = False, backbone = 'vgg', is_body_batch=False,
                 enc_batch = False, dec_batch = False,  vgg_cfg_ind ='D', fu_func = 'cat', up_func = 'bl', is_AG=False, skip_attention = 'none'):
        super(Unet, self).__init__()
        if is_body_batch:
            enc_batch = True
            dec_batch = True
        self.backbone = backbone
        self.is_AG = is_AG
        self.fu_func = fu_func
        self.up_func = up_func
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained = pretrained, batch_norm=enc_batch, cfg_ind= vgg_cfg_ind)
            in_filters = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            in_filters  = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3], is_batch=dec_batch, is_AG=self.is_AG, fu_func=fu_func, up_func= self.up_func, skip_attention=skip_attention)
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2], is_batch=dec_batch, is_AG=self.is_AG, fu_func=fu_func, up_func= self.up_func, skip_attention=skip_attention)
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1], is_batch=dec_batch, is_AG=self.is_AG, fu_func=fu_func, up_func= self.up_func, skip_attention=skip_attention)
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0], is_batch=dec_batch, is_AG=self.is_AG, fu_func=fu_func, up_func= self.up_func, skip_attention=skip_attention)

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)
        
        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
