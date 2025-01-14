import json
import torch
from nets.deeplab.deeplabv3_plus import DeepLab
import tqdm
import loss
import data_set
import nets.unet
from torch.utils.data import Dataset, DataLoader
import assess.process_data
import file_pro.json_pro
import create_model


def acc(net_name, weights_path, out_json_path):

    test_set_ = data_set.set('dataset/img',
                             'dataset/crater',
                             'dataset/ejecta',
                             'dataset/test_list_ag.txt')

    train_net = create_model.create(net_name)

    test_dic = {
        'net': net_name,
        'weights': weights_path,
        'ious': {},
    }

    train_net.load_state_dict(torch.load(weights_path))
    train_net.eval()
    with torch.no_grad():
        print(assess.process_data.get_all(
            test_set_, train_net, is_log=True, test_dic=test_dic))
    file_pro.json_pro.write_json(test_dic, out_json_path)

