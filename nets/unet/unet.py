import torch
import torch.nn as nn

from nets.unet.vgg import VGG16


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
    def __init__(self, in_size, out_size, skip_x_size, x_size, is_attention = True, is_batch = False):
        super(unetUp, self).__init__()
        self.is_attention = is_attention
        if is_attention:
            self.upsample = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
            self.attention = AttentionBlock(skip_x_size, x_size, int((in_size - out_size) / 2), is_batch)
            if is_batch:
                self.conv_bn1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
                                              nn.BatchNorm2d(out_size))
                self.conv_bn2 = nn.Sequential(nn.Conv2d(out_size, out_size, kernel_size=3, padding=1),
                                              nn.BatchNorm2d(out_size))
            else:
                self.conv_bn1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
                self.conv_bn2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        else:
            self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
            self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
            self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
            self.relu   = nn.ReLU(inplace = True)

    def forward(self, skip, x):
        if self.is_attention:
            x_attention = self.attention(skip, x)
            x = nn.functional.interpolate(x, skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat((x_attention, x), dim=1)
            x = self.conv_bn1(x)
            return self.conv_bn2(x)
        else:
            outputs = torch.cat([skip, self.up(x)], 1)
            outputs = self.conv1(outputs)
            outputs = self.relu(outputs)
            outputs = self.conv2(outputs)
            outputs = self.relu(outputs)
            return outputs


class Unet(nn.Module):
    def __init__(self, num_classes = 21, pretrained = False, backbone = 'vgg', is_attention = False, is_batch = False, is_body_batch=False, vgg_cfg_ind = 'D'):
        super(Unet, self).__init__()
        self.backbone = backbone
        self.vgg = VGG16(pretrained = pretrained, batch_norm=is_body_batch, cfg_ind= vgg_cfg_ind)
        in_filters = [192, 384, 768, 1024]
        x_channels = [128, 256, 512, 512]
        skip_x_channels = [64, 128, 256, 512]
        out_filters = [64, 128, 256, 512]

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3], skip_x_channels[3], x_channels[3], is_attention, is_batch)
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2], skip_x_channels[2], x_channels[2], is_attention, is_batch)
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1], skip_x_channels[1], x_channels[1], is_attention, is_batch)
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0], skip_x_channels[0], x_channels[0], is_attention, is_batch)
        self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)
        
        return final

    def freeze_backbone(self):
        for param in self.vgg.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.vgg.parameters():
            param.requires_grad = True
