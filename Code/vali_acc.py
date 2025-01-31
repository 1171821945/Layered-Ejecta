import json
import torch
import create_model
from nets.deeplab.deeplabv3_plus import DeepLab
import tqdm
import loss
import data_set
import nets.unet
from torch.utils.data import Dataset, DataLoader
import assess.process_data
import file_pro.json_pro


def acc(net_name):
    L_d = 'L1'
    vali_data_path = 'data/{}/vali_data.json'.format(net_name)
    weights_init_path = 'data/{}/init.pth'.format(net_name)
    weights_paths = [weights_init_path]
    for i in range(1, 21):
        weights_paths.append('data/{}/step_weights/train_net{}.pth'.format(net_name, i * 5))
    vali_set = data_set.set('dataset/img',
                            'dataset/crater',
                            'dataset/ejecta',
                            'dataset/vali_list_ag.txt.txt')
    print(weights_paths)

    train_net = create_model.create(net_name)

    vali_dic = {
        'net': net_name,
        'loss_vali': [],
        'con_mat_vali': [],
        'iou_vali': [],
        'precision_vali': [],
        'recall_vali': [],
        'score_vali': [],
    }

    for weights_path in weights_paths:
        train_net.load_state_dict(torch.load(weights_path))
        train_net.eval()
        with torch.no_grad():
            loss_vali, con_mat_vali, iou_vali, precision_vali, recall_vali, score1_vali, score2_vali = assess.process_data.get_all(
                vali_set, train_net)
            vali_dic['loss_vali'].append(loss_vali)
            vali_dic['con_mat_vali'].append(con_mat_vali)
            vali_dic['iou_vali'].append(iou_vali)
            vali_dic['precision_vali'].append(precision_vali)
            vali_dic['recall_vali'].append(recall_vali)
            vali_dic['score_vali'].append([score1_vali, score2_vali])
        file_pro.json_pro.write_json(vali_dic, vali_data_path)



