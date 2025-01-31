import vali_acc
import test_acc
import batch_predict
import os


if __name__ == '__main__':
    # net_name =
    # ema_unet(g=4) ema_unet(g=8) ema_unet(g=16)
    # eucb_ema_unet(g=4) eucb_ema_unet(g=8) eucb_ema_unet(g=16)
    # eucb_unet mhema_unet
    # emhseu_unet
    # res_unet attention_unet deeplabv3+ pspnet fcn segnet
    net_name = 'emhseu_unet'


    for path2 in ['best_iou.pth']:
        dir1 = 'data/{}/test_data/{}/'.format(net_name, path2[:-4])
        test_acc.acc(net_name, dir1 + path2, 'data/{}/test_data/'.format(net_name) + path2[:-4] + '/test_data.json')





