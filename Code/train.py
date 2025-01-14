import json
import os.path

from torch.cuda.amp import GradScaler

import file_pro.json_pro
import torch
import tqdm
import loss
import data_set
from torch.utils.data import Dataset, DataLoader
import assess.process_data
import create_model

# Learning rate related parameters
init_lr = 5e-4
min_lr = 7e-7
aph = 0.95
lr = init_lr



net_name = 'emhseu_unet'

# Whether to load data
is_load_data = False
is_im_loss = False

if not is_load_data and os.path.isdir('data/{}/step_weights'.format(net_name)):
    raise Exception

data_path = 'data/{}/data.json'.format(net_name)
weights_path = 'data/{}/train_net.pth'.format(net_name)
weights_step_dir = 'data/{}/step_weights'.format(net_name)
weights_init_path = 'data/{}/init.pth'.format(net_name)
if not os.path.isdir(weights_step_dir):
    os.makedirs(weights_step_dir)


echops = 100

train_set_ = data_set.set('dataset/img',
                         'dataset/crater',
                         'dataset/ejecta',
                         'dataset/train_list.txt')

vali_set_ = data_set.set('dataset/img',
                         'dataset/crater',
                         'dataset/ejecta',
                         'dataset/vali_list_ag.txt')
train_set = DataLoader(train_set_, 32)
# Create a network model
train_net = create_model.create(net_name)
print('Network model creation completed')
if is_load_data:
    train_net.load_state_dict(torch.load(weights_path))
    print('Imported saved network')
else:
    torch.save(train_net.state_dict(), weights_init_path)
    print('The initial network has been saved')
# loss function
train_loss = loss.CE_loss(imloss = is_im_loss).cuda()

data = {
    'times': -1,
    'net': net_name,
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
if is_load_data:
    with open(data_path) as f:
        data = f.read()
        data = json.loads(data)
        lr = data['lr'][-1]*aph


def get_lr(lr0):
    if lr0*aph > min_lr:
        lr_ = lr0*aph
    else:
        lr_ = min_lr
    return lr_


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
    print(data)
    with open(data_path, 'w') as f:
        f.write(json.dumps(data, indent=1))


# train
def fit(model, lr, echops):
    model.train()
    if is_load_data:
        best_value = file_pro.json_pro.read_json('data/{}/best_value.json'.format(net_name))
    else:
        best_value = {
            'best_iou': 0,
            'best_loss': 10,
            'best_f1': 0
        }

    def get_sto(train_loss):
        model.eval()
        # Evaluate the training and validation sets
        with torch.no_grad():
            # loss_train, con_mat_train, iou_train, precision_train, recall_train, score1_train, score2_train = assess.process_data.get_all(
            #     train_set_, model)
            loss_vali, con_mat_vali, iou_vali, precision_vali, recall_vali, score1_vali, score2_vali = assess.process_data.get_all(
                vali_set_, model, is_imp_loss=is_im_loss)
            loss_train, con_mat_train, iou_train, precision_train, recall_train, score1_train, score2_train = train_loss, 0, 0, 0, 0, 0, 0
            for d in ['best_iou', 'best_f1', 'best_loss']:
                if not os.path.isdir('data/{}/test_data/{}'.format(net_name, d)):
                    os.makedirs('data/{}/test_data/{}'.format(net_name, d))

            if iou_vali > best_value['best_iou']:
                best_value['best_iou'] = iou_vali
                torch.save(model.state_dict(), 'data/{}/test_data/best_iou/best_iou.pth'.format(net_name))
            if score1_vali > best_value['best_f1']:
                best_value['best_f1'] = score1_vali
                torch.save(model.state_dict(), 'data/{}/test_data/best_f1/best_f1.pth'.format(net_name))
            if loss_vali < best_value['best_loss']:
                best_value['best_loss'] = loss_vali
                torch.save(model.state_dict(), 'data/{}/test_data/best_loss/best_loss.pth'.format(net_name))
            file_pro.json_pro.write_json(best_value, 'data/{}/best_value.json'.format(net_name))

        sto_data(lr, loss_train, loss_vali, con_mat_train, con_mat_vali,
                 iou_train, iou_vali, precision_train, precision_vali,
                 recall_train, recall_vali, [score1_train, score2_train], [score1_vali, score2_vali])
        model.train()
    if not is_load_data:
        get_sto(0)
    for echop in range(echops):
  #      train_set = DataLoader(train_set_, 32, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr)
        print("Start the {} round of training".format(echop))
        with tqdm.tqdm(total=len(train_set)) as t:
            loss_value = 0
            for ind, (x, y, _) in enumerate(train_set):
                x = x.cuda()
                y = y.cuda()
                y_pre = model.forward(x)
                loss_tensor = train_loss(y_pre, y)
                loss_value = loss_value+loss_tensor
                t.set_postfix(loss=loss_value/(ind+1))
                loss_tensor.backward()
                opt.step()
                opt.zero_grad()
                t.update(1)
        print("The {} round of training is completed, and the weights will be automatically saved".format(echop))
        torch.save(model.state_dict(), weights_path)
        print("Automatic weight saving completed")
        get_sto(float(loss_value/(ind+1)))
        if data['times'] % 5 == 0:
            torch.save(train_net.state_dict(), '{}/train_net{}.pth'.format(weights_step_dir, data['times']))
        lr = get_lr(lr)

def fit_mixpre(model, lr, echops, mix_pre = False):
    model.train()
    if is_load_data:
        best_value = file_pro.json_pro.read_json('data/{}/best_value.json'.format(net_name))
    else:
        best_value = {
            'best_iou': 0,
            'best_loss': 10,
            'best_f1': 0
        }

    def get_sto():
        model.eval()
        # Evaluate the training and validation sets
        with torch.no_grad():
            # loss_train, con_mat_train, iou_train, precision_train, recall_train, score1_train, score2_train = assess.process_data.get_all(
            #     train_set_, model)
            loss_vali, con_mat_vali, iou_vali, precision_vali, recall_vali, score1_vali, score2_vali = assess.process_data.get_all(
                vali_set_, model, is_imp_loss=is_im_loss)
            loss_train, con_mat_train, iou_train, precision_train, recall_train, score1_train, score2_train = 0, 0, 0, 0, 0, 0, 0
            for d in ['best_iou', 'best_f1', 'best_loss']:
                if not os.path.isdir('data/{}/test_data/{}'.format(net_name, d)):
                    os.makedirs('data/{}/test_data/{}'.format(net_name, d))

            if iou_vali > best_value['best_iou']:
                best_value['best_iou'] = iou_vali
                torch.save(model.state_dict(), 'data/{}/test_data/best_iou/best_iou.pth'.format(net_name))
            if score1_vali > best_value['best_f1']:
                best_value['best_f1'] = score1_vali
                torch.save(model.state_dict(), 'data/{}/test_data/best_f1/best_f1.pth'.format(net_name))
            if loss_vali < best_value['best_loss']:
                best_value['best_loss'] = loss_vali
                torch.save(model.state_dict(), 'data/{}/test_data/best_loss/best_loss.pth'.format(net_name))
            file_pro.json_pro.write_json(best_value, 'data/{}/best_value.json'.format(net_name))
            # loss_vali, con_mat_vali, iou_vali, precision_vali, recall_vali, score1_vali, score2_vali = 0, 0, 0, 0, 0, 0, 0

        sto_data(lr, loss_train, loss_vali, con_mat_train, con_mat_vali,
                 iou_train, iou_vali, precision_train, precision_vali,
                 recall_train, recall_vali, [score1_train, score2_train], [score1_vali, score2_vali])
        model.train()
    if not is_load_data:
        get_sto()
    if mix_pre:
        scaler = GradScaler()
    for echop in range(echops):
  #      train_set = DataLoader(train_set_, 32, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr)
        print("Start the {} round of training".format(echop))
        with tqdm.tqdm(total=len(train_set)) as t:
            loss_value = 0
            for ind, (x, y, _) in enumerate(train_set):
                x = x.cuda()
                y = y.cuda()
                if mix_pre:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        y_pre = model.forward(x)
                        loss_tensor = train_loss(y_pre, y)
                else:
                    y_pre = model.forward(x)
                    loss_tensor = train_loss(y_pre, y)
                loss_value = loss_value+loss_tensor
                t.set_postfix(loss=loss_value/(ind+1))
                if mix_pre:
                    scaler.scale(loss_tensor).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss_tensor.backward()
                    opt.step()

                opt.zero_grad()
                t.update(1)
        print("The {} round of training is completed, and the weights will be automatically saved".format(echop))
        torch.save(model.state_dict(), weights_path)
        print("Automatic weight saving completed")
        get_sto()
        if data['times'] % 5 == 0:
            torch.save(train_net.state_dict(), '{}/train_net{}.pth'.format(weights_step_dir, data['times']))
        lr = get_lr(lr)


def fit_accsteps(model, lr, echops, is_acc_steps = False):
    if is_acc_steps:
        acc_steps = 2
    model.train()
    if is_load_data:
        best_value = file_pro.json_pro.read_json('data/{}/best_value.json'.format(net_name))
    else:
        best_value = {
            'best_iou': 0,
            'best_loss': 10,
            'best_f1': 0
        }

    def get_sto():
        model.eval()
        # Evaluate the training and validation sets
        with torch.no_grad():
            # loss_train, con_mat_train, iou_train, precision_train, recall_train, score1_train, score2_train = assess.process_data.get_all(
            #     train_set_, model)
            loss_vali, con_mat_vali, iou_vali, precision_vali, recall_vali, score1_vali, score2_vali = assess.process_data.get_all(
                vali_set_, model, is_imp_loss=is_im_loss)
            loss_train, con_mat_train, iou_train, precision_train, recall_train, score1_train, score2_train = 0, 0, 0, 0, 0, 0, 0
            for d in ['best_iou', 'best_f1', 'best_loss']:
                if not os.path.isdir('data/{}/test_data/{}'.format(net_name, d)):
                    os.makedirs('data/{}/test_data/{}'.format(net_name, d))

            if iou_vali > best_value['best_iou']:
                best_value['best_iou'] = iou_vali
                torch.save(model.state_dict(), 'data/{}/test_data/best_iou/best_iou.pth'.format(net_name))
            if score1_vali > best_value['best_f1']:
                best_value['best_f1'] = score1_vali
                torch.save(model.state_dict(), 'data/{}/test_data/best_f1/best_f1.pth'.format(net_name))
            if loss_vali < best_value['best_loss']:
                best_value['best_loss'] = loss_vali
                torch.save(model.state_dict(), 'data/{}/test_data/best_loss/best_loss.pth'.format(net_name))
            file_pro.json_pro.write_json(best_value, 'data/{}/best_value.json'.format(net_name))
            # loss_vali, con_mat_vali, iou_vali, precision_vali, recall_vali, score1_vali, score2_vali = 0, 0, 0, 0, 0, 0, 0

        sto_data(lr, loss_train, loss_vali, con_mat_train, con_mat_vali,
                 iou_train, iou_vali, precision_train, precision_vali,
                 recall_train, recall_vali, [score1_train, score2_train], [score1_vali, score2_vali])
        model.train()
    if not is_load_data:
        get_sto()
    for echop in range(echops):
        opt = torch.optim.Adam(model.parameters(), lr)
        print("Start the {} round of training".format(echop))
        with tqdm.tqdm(total=(len(train_set)+1)//acc_steps) as t:
            loss_value = 0
            for ind, (x, y, _) in enumerate(train_set):
                x = x.cuda()
                y = y.cuda()
                y_pre = model.forward(x)
                loss_tensor = train_loss(y_pre, y)
                loss_value = loss_value+loss_tensor    
                loss_tensor.backward()
                if (ind+1)%acc_steps == 0 or (ind+1) == len(train_set):
                    t.set_postfix(loss=loss_value/(ind+1))
                    opt.step()
                    opt.zero_grad()
                    t.update(1)             
        print("The {} round of training is completed, and the weights will be automatically saved".format(echop))
        torch.save(model.state_dict(), weights_path)
        print("Automatic weight saving completed")
        get_sto()
        if data['times'] % 5 == 0:
            torch.save(train_net.state_dict(), '{}/train_net{}.pth'.format(weights_step_dir, data['times']))
        lr = get_lr(lr)

fit(train_net, lr, echops)


