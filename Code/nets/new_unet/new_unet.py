import torch
import torch.nn as nn
from attention.ema.ema import EMA, MHEMA
from nets.new_unet.nu_vgg import VGG16
from modules.EUCB import EUCB



class unetUp(nn.Module):
    def __init__(self, in_size, out_size, up_func = 'bl'):
        super(unetUp, self).__init__()


        conv1_layers = [nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)]



        conv2_layers = [nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)]
        conv1_layers += [nn.ReLU(inplace=True)]
        conv2_layers += [nn.ReLU(inplace=True)]
        self.conv1 = nn.Sequential(*conv1_layers)
        self.conv2 = nn.Sequential(*conv2_layers)
        if up_func == 'bl':
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        elif up_func == 'eucb':
            self.up = EUCB(in_size - out_size, in_size - out_size)
        elif up_func == 'tc':
            self.up = nn.ConvTranspose2d(in_size - out_size, in_size - out_size, 3, 2, 1, 1)
        elif up_func == 'nn':
            self.up = nn.Upsample(scale_factor=2)


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
        elif self.fu_func == 'cafm':
            x = self.cafm_conv(self.up(x))
            outputs1, outputs2 = self.CAFM(skip, x)
            outputs = outputs1 + outputs2
        else :
            raise Exception
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)

        return outputs



class Unet(nn.Module):
    # isbatch: if bn?
    # is_body_batch net(VGG)
    def __init__(self, num_classes = 21, pretrained = False, backbone = 'vgg', vgg_cfg_ind ='D', up_func = 'bl'):
        super(Unet, self).__init__()
        self.backbone = backbone

        self.up_func = up_func
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained = pretrained, batch_norm=enc_batch, cfg_ind= vgg_cfg_ind)
            in_filters = [192, 384, 768, 1024]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3], up_func= self.up_func)
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2], up_func= self.up_func)
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1], up_func= self.up_func)
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0], up_func= self.up_funcn)

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        else:
            raise Exception

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)
        
        return final
