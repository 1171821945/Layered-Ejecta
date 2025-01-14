import torch.nn as nn
from torch.hub import load_state_dict_from_url
from attention.ema.ema import EMA, MHEMA, MHEMA_sum, MHEMA_sum2, MHEMA2
from attention.aa.aa import AgentAttention
from attention.caa.caa import CAA
from modules.down_wt import Down_wt
from modules.wtconv import DepthwiseSeparableConvWithWTConv2d,WTConv2d
from modules.CAFM import LinAngularXCA_CA
from modules.CAFM2 import Attention as CAFM2
from modules.DWConv import SeparableConv2d as DWConv
from modules.ASPP import ASPP
import modules
class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features, self.down_index = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        # x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        feat1 = self.features[  :self.down_index[0]](x)
        feat2 = self.features[self.down_index[0]:self.down_index[1] ](feat1)
        feat3 = self.features[self.down_index[1] :self.down_index[2]](feat2)
        feat4 = self.features[self.down_index[2]:self.down_index[3]](feat3)
        feat5 = self.features[self.down_index[3]:-1](feat4)
        return [feat1, feat2, feat3, feat4, feat5]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, in_channels = 3):
    last_layer_c = 0
    layers = []
    down_index = []
    for v in cfg:
        if v == 'M':
            down_index.append(len(layers))
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'CD':
            down_index.append(len(layers))
            layers += [nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2)]
        elif v == 'DW':
            down_index.append(len(layers))
            layers += [Down_wt(last_layer_c, last_layer_c)]
        elif v == 'DWC_DS':
            down_index.append(len(layers))
            layers += [DWConv(last_layer_c, last_layer_c, 2, 2)]
        elif v == 'AD':
            down_index.append(len(layers))
            layers += [Down_wt(last_layer_c, last_layer_c)]


        elif v == 'EMA':
            layers += [EMA(last_layer_c)]
        elif v == 'CAA':
            layers += [CAA(last_layer_c)]
        elif v == 'MHEMA':
            layers += [MHEMA(last_layer_c)]
        elif v == 'MHEMA2':
            layers += [MHEMA2(last_layer_c)]
        elif v == 'MHSEMA':
            layers += [MHEMA_sum(last_layer_c)]
        elif v == 'MHSEMA2':
            layers += [MHEMA_sum2(last_layer_c)]
        elif v == 'AA':
            layers += [AgentAttention(last_layer_c, 32*32)]
        elif v == 'CAFM0':
            layers += [LinAngularXCA_CA()]
        elif v == 'CAFM2':
            layers += [CAFM2(last_layer_c)]
        elif v == 'ASPP':
            layers += [ASPP(last_layer_c, last_layer_c, (6, 12, 18))]


        else:
            if type(v) == str:
                if v[0:2] == 'WT':
                    v = int(v[2:])
                    conv2d = WTConv2d(in_channels, v)
                elif v[0:3] == 'DWC':
                    v = int(v[3:])
                    conv2d = DWConv(in_channels, v, 3, 1, 1)
                else:
                    raise
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)

            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]

            last_layer_c = v
            in_channels = v
    return nn.Sequential(*layers), down_index
# 512,512,3 -> 512,512,64 -> 256,256,64 -> 256,256,128 -> 128,128,128 -> 128,128,256 -> 64,64,256
# 64,64,512 -> 32,32,512 -> 32,32,512
cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'I': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'CD', 512, 512, 512, 'CD', 512, 512, 512, 'M'],
    'A': [64, 64, 'CD', 128, 128, 'CD', 256, 256, 256, 'CD', 512, 512, 512, 'CD', 512, 512, 512, 'M'],
    'MS0': [64, 64, 'M', 'MS', 128, 128,  'M', 'MS', 256, 256, 256, 'M', 'MS', 512, 512, 512, 'M', 'MS', 512, 512, 512, 'M'],# multy scale head
    'EMA0': [64, 64, 'M', 'EMA', 128, 128,  'M', 'EMA', 256, 256, 256, 'M', 'EMA', 512, 512, 512, 'M', 'EMA', 512, 512, 512, 'M'],
    'EMA1': [64, 'EMA', 64, 'M', 128, 'EMA',  128,  'M', 256, 256, 'EMA', 256, 'M', 512, 512, 'EMA', 512, 'M', 512, 512, 512, 'M'],
    'EMA2': [64, 64, 'EMA', 'M', 128,  128, 'EMA',  'M', 256, 256, 256, 'EMA', 'M', 512, 512, 512, 'EMA', 'M', 512, 512, 512, 'M'],
    'MHEMA0': [64, 64, 'M', 'MHEMA', 128, 128,  'M', 'MHEMA', 256, 256, 256, 'M', 'MHEMA', 512, 512, 512, 'M', 'MHEMA', 512, 512, 512, 'M'],
    'MHEMA20': [64, 64, 'M', 'MHEMA2', 128, 128,  'M', 'MHEMA2', 256, 256, 256, 'M', 'MHEMA2', 512, 512, 512, 'M', 'MHEMA2', 512, 512, 512, 'M'],
    'MHEMA(skip)': [64, 64, 'MHEMA', 'M', 128, 128,  'MHEMA', 'M', 256, 256, 256, 'MHEMA', 'M', 512, 512, 512, 'MHEMA', 'M', 512, 512, 512, 'M'],
    'MHSEMA0': [64, 64, 'M', 'MHSEMA', 128, 128,  'M', 'MHSEMA', 256, 256, 256, 'M', 'MHSEMA', 512, 512, 512, 'M', 'MHSEMA', 512, 512, 512, 'M'],
    'MHSEMA2': [64, 64, 'MHSEMA2', 'M', 128,  128, 'MHSEMA2',  'M', 256, 256, 256, 'MHSEMA2', 'M', 512, 512, 512, 'MHSEMA2', 'M', 512, 512, 512, 'M'],
    'DWMHEMA0': [64, 64, 'DW', 'MHEMA', 128, 128,  'DW', 'MHEMA', 256, 256, 256, 'DW', 'MHEMA', 512, 512, 512, 'DW', 'MHEMA', 512, 512, 512, 'M'],
    'CDEMA0': [64, 64, 'CD', 'EMA', 128, 128,  'CD', 'EMA', 256, 256, 256, 'CD', 'EMA', 512, 512, 512, 'CD', 'EMA', 512, 512, 512, 'M'],
    'CDMHEMA0': [64, 64, 'CD', 'MHEMA', 128, 128,  'CD', 'MHEMA', 256, 256, 256, 'CD', 'MHEMA', 512, 512, 512, 'CD', 'MHEMA', 512, 512, 512, 'M'],
    'DWEMA0': [64, 64, 'DW', 'EMA', 128, 128,  'DW', 'EMA', 256, 256, 256, 'DW', 'EMA', 512, 512, 512, 'DW', 'EMA', 512, 512, 512, 'M'],
    'ADEMA0': [64, 64, 'AD', 'EMA', 128, 128,  'AD', 'EMA', 256, 256, 256, 'AD', 'EMA', 512, 512, 512, 'AD', 'EMA', 512, 512, 512, 'M'],
    'WT_EMA0': ['WT64', 'WT64', 'M', 'EMA', 'WT128', 'WT128',  'M', 'EMA', 'WT256', 'WT256', 'WT256', 'M', 'EMA', 'WT512', 'WT512', 'WT512', 'M', 'EMA','WT512','WT512', 'WT512', 'M'],
    'WT_EMA1': [64, 'WT64', 'M', 'EMA', 128, 'WT128',  'M', 'EMA', 256, 256, 'WT256', 'M', 'EMA', 512, 512, 'WT512', 'M', 'EMA',512, 512, 'WT512', 'M'],
    'CAA0': [64, 64, 'M', 'CAA', 128, 128,  'M', 'CAA', 256, 256, 256, 'M', 'CAA', 512, 512, 512, 'M', 'CAA', 512, 512, 512, 'M'],
    'CAFM_EMA00': [64, 64, 'M', 'EMA', 128, 128,  'M', 'EMA', 256, 256, 256, 'M', 'EMA', 512, 512, 512, 'M', 'EMA', 512, 512, 512, 'CAFM0', 'M'],
    'AA0': [64, 64, 'M', 128, 128,  'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'AA', 'M'],
    'AA_EMA0': [64, 64, 'M', 'EMA', 128, 128,  'M', 'EMA', 256, 256, 256, 'M', 'EMA', 512, 512, 512, 'M', 'EMA', 512, 512, 512, 'AA', 'M'],
    'FUll_DWC_EMA0': ['DWC64', 'DWC64', 'DWC_DS', 'EMA', 'DWC128', 'DWC128',  'DWC_DS', 'EMA', 'DWC256', 'DWC256', 'DWC256', 'DWC_DS', 'EMA', 'DWC512', 'DWC512', 'DWC512', 'DWC_DS', 'EMA', 'DWC512', 'DWC512', 'DWC512', 'M'],
    'DWCDS_EMA0': [64, 64, 'DWC_DS', 'EMA', 128, 128,  'DWC_DS', 'EMA', 256, 256, 256, 'DWC_DS', 'EMA', 512, 512, 512, 'DWC_DS', 'EMA', 512, 512, 512, 'M'],
    'DWC_EMA0': ['DWC64', 'DWC64', 'M', 'EMA', 'DWC128', 'DWC128',  'M', 'EMA', 'DWC256', 'DWC256', 'DWC256', 'M', 'EMA', 'DWC512', 'DWC512', 'DWC512', 'M', 'EMA', 'DWC512', 'DWC512', 'DWC512', 'M'],
    'DWCDS_MHEMA0': [64, 64, 'DWC_DS', 'MHEMA', 128, 128,  'DWC_DS', 'MHEMA', 256, 256, 256, 'DWC_DS', 'MHEMA', 512, 512, 512, 'DWC_DS', 'MHEMA', 512, 512, 512, 'M'],
    'ASPP_EMA0':[64, 64, 'M', 'EMA', 128, 128,  'M', 'EMA', 256, 256, 256, 'M', 'EMA', 512, 512, 512, 'M', 'EMA', 512, 512, 512, 'ASPP', 'M'],
    'ASPP_EMA2': [64, 64, 'EMA', 'M', 128,  128, 'EMA',  'M', 256, 256, 256, 'EMA', 'M', 512, 512, 512, 'EMA', 'M', 512, 512, 512, 'ASPP', 'M'],
    'ASPP0':[64, 64, 'M', 128, 128,  'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'ASPP', 'M'],
    'CAFM2_EMA0': [64, 64, 'M', 'EMA', 128, 128,  'M', 'EMA', 256, 256, 256, 'M', 'EMA', 512, 512, 512, 'M', 'EMA', 512, 512, 512, 'CAFM2', 'M'],
    # 'EMA1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}


def VGG16(pretrained, in_channels = 4, batch_norm = False, cfg_ind = None, **kwargs):
    model = VGG(make_layers(cfgs[cfg_ind], batch_norm = batch_norm, in_channels = in_channels), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth", model_dir="./model_data")
        model.load_state_dict(state_dict)
    
    del model.avgpool
    del model.classifier
    return model
