import json
import torch
import tqdm
import loss
import data_set
from torch.utils.data import DataLoader
import assess.process_data
import create_model

# Learning rate related parameters
init_lr = 5e-4
min_lr = 7e-7
aph = 0.95
lr = init_lr


# if load data to continue training
is_load_data = False

data_path = 'data/data.json'
weights_path = 'data/train_net.pth'
weights_step_dir = 'data/step_weights'
weights_init_path = 'data/init.pth'

echops = 100
batch_size = 32
is_load_data = False

train_set_ = data_set.set('dataset/img',
                         'dataset/crater',
                         'dataset/ejecta',
                         'dataset/train_list.txt')

vali_set_ = data_set.set('dataset/img',
                         'dataset/crater',
                         'dataset/ejecta',
                         'dataset/vali_list.txt')
train_set = DataLoader(train_set_, batch_size)
# model
train_net = create_model.create()
print('Network creation completed')
if is_load_data:
    train_net.load_state_dict(torch.load(weights_path))
    print('Imported saved network')
else:
    torch.save(train_net.state_dict(), weights_init_path)
    print('The initial network has been saved')

# loss function
train_loss = loss.EC_loss().cuda()
data = {
    'times': -1,
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


# Train the network
def fit(model, lr, echops):
    def get_sto():
        model.eval()
        # Evaluate the training and validation sets
        with torch.no_grad():
            loss_train, con_mat_train, iou_train, precision_train, recall_train, score1_train, score2_train = assess.process_data.get_all(
                train_set_, model)
            loss_vali, con_mat_vali, iou_vali, precision_vali, recall_vali, score1_vali, score2_vali = assess.process_data.get_all(
                vali_set_, model)

        sto_data(lr, loss_train, loss_vali, con_mat_train, con_mat_vali,
                 iou_train, iou_vali, precision_train, precision_vali,
                 recall_train, recall_vali, [score1_train, score2_train], [score1_vali, score2_vali])
    if not is_load_data:
        get_sto()
    for echop in range(echops):
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr)
        print("Start the {}-th round of training".format(echop))
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
        print("The {}-th round of training is completed, and the weights will be automatically saved".format(echop))
        torch.save(model.state_dict(), weights_path)
        print("Auto save weight completed")
        get_sto()
        if data['times'] % 5 == 0:
            torch.save(train_net.state_dict(), '{}/train_net{}.pth'.format(weights_step_dir, data['times']))
        lr = get_lr(lr)


fit(train_net, train_set, lr, echops)


