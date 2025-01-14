from nets.new_unet import new_unet

def create(net_name):
    if net_name == 'emhseu_unet':
        train_net = new_unet.Unet(num_classes=2, vgg_cfg_ind='MHEMA20', up_func='eucb').cuda()