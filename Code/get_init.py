import json
import torch
import data_set
import assess.process_data

data_path = 'data/data_resnet/data_resnet_init.json'
weights_path = 'data/data_resnet/init.pth'

train_set_ = data_set.set('enhance/img',
                         'enhance/crater',
                         'enhance/ejecta',
                         'enhance/train_list.txt')

vali_set_ = data_set.set('enhance/img',
                         'enhance/crater',
                         'enhance/ejecta',
                         'enhance/vali_list.txt')

# 网络模型
train_net = nets.unet.Unet(num_classes=2, backbone='resnet50').cuda()
train_net.load_state_dict(torch.load(weights_path))

data = {
    'times': 0,
    'loss_train': [],
    'loss_vali': [],
    'con_mat_train': [],
    'con_mat_vali': [],
    'iou_train': [],
    'iou_vali': [],
    'precision_train': [],
    'precision_vali': [],
    'recall_train': [],
    'recall_vali': [],
    'score_train': [],
    'score_vali': [],
    'lr': []
}


def sto_data(lr, loss_train, loss_vali, con_mat_train,
             con_mat_vali, iou_train, iou_vali, precision_train,
             precision_vali, recall_train, recall_vali, score_train, score_vali):
    data['times'] += 1
    data['lr'].append(lr)
    data['loss_train'].append(loss_train)
    data['loss_vali'].append(loss_vali)
    data['con_mat_train'].append(con_mat_train)
    data['con_mat_vali'].append(con_mat_vali)
    data['iou_train'].append(iou_train)
    data['iou_vali'].append(iou_vali)
    data['precision_train'].append(precision_train)
    data['precision_vali'].append(precision_vali)
    data['recall_train'].append(recall_train)
    data['recall_vali'].append(recall_vali)
    data['score_train'].append(score_train)
    data['score_vali'].append(score_vali)
    with open(data_path, 'w') as f:
        f.write(json.dumps(data, indent=1))


train_net.eval()
with torch.no_grad():
        loss_train, con_mat_train, iou_train, presision_train, recall_train, score1_train, score2_train = assess.process_data.get_all(train_set_, train_net)
        loss_vali, con_mat_vali, iou_vali, presision_vali, recall_vali, score1_vali, score2_vali = assess.process_data.get_all(vali_set_, train_net)

sto_data(0, loss_train, loss_vali, con_mat_train, con_mat_vali,
                 iou_train, iou_vali, presision_train, presision_vali,
                 recall_train, recall_vali, [score1_train, score2_train], [score1_vali, score2_vali])





