import torch
from PIL import Image
import numpy as np
import os
import nets.unet
import img_pro
import data_set
from nets.deeplab.deeplabv3_plus import DeepLab
import create_model


def blend(img, mask, colors):
    alpha = np.array(mask)
    alpha = np.stack([alpha, alpha, alpha, alpha], -1)
    alpha = Image.fromarray(alpha)
    img = img.convert('RGBA')
    mask = np.array(colors, np.uint8)[np.array(mask)]
    mask = Image.fromarray(mask).convert('RGBA')
    blend_img = Image.composite(mask, img, alpha.point(lambda i: 64 if i > 0 else 0))
    return blend_img


def batch_predict(net_name, weights_path, save_dir):
    if not os.path.isdir(save_dir + '/prd'):
        os.makedirs(save_dir + '/prd')
    list_path = 'dataset/test_list_ag.txt'

    colors = [(0, 0, 0), (188, 10, 10)]

    x1_dir = 'dataset/img/'
    x2_dir = 'dataset/crater/'

    train_net = create_model.create(net_name)

    model = train_net
    model.load_state_dict(torch.load(weights_path))

    def predict(model, x1_path, x2_path):
        x1 = Image.open(x1_path)
        x2 = Image.open(x2_path)
        x1 = img_pro.resize(x1, 512, 512)
        x2 = img_pro.resize(x2, 512, 512)
        x1_ = x1
        x2_ = x2
        x1 = np.array(x1, np.float32)
        x2 = np.array(x2, np.float32)
        x1 = np.transpose(x1, [2, 0, 1]) / 255.
        x2 = np.expand_dims(x2, 0)
        x = np.concatenate([x1, x2], 0)
        x = np.expand_dims(x, 0)
        x = torch.from_numpy(x).cuda()
        model.eval()
        with torch.no_grad():
            y = model(x)[0]
        y = y.permute([1, 2, 0])
        return x1_, x2_, y

    def get_predict(img_name):
        x1_path = x1_dir + img_name + '.jpg'
        x2_path = x2_dir + img_name + '.png'
        x1, x2, y = predict(model, x1_path, x2_path)
        y = torch.nn.functional.softmax(y, dim=-1)
        y = torch.argmax(y, -1).cpu()
        y = Image.fromarray(np.array(y, np.uint8))
        x_y = blend(x1, y, colors)
        x_y.save(save_dir + '/prd/' + img_name + '.png')
        return y, x_y

    with open(list_path) as f:
        img_names = f.readlines()
    for i in range(len(img_names)):
        img_names[i] = img_names[i].strip()

    for i in img_names:
        print(i)
        get_predict(i)


def batch_predict_(net_name, list_path, x1_dir, x2_dir, predict_img_save_dir, is_best=True):
    L_d = 'L1'
    if is_best:
        D_d = 'best'
    else:
        D_d='100'
    weights_path = 'data/{}/{}/{}/train_net.pth'.format(net_name, L_d, D_d)

    colors = [(0, 0, 0), (64, 0, 0)]

    train_net = create_model.create(net_name)

    model = train_net
    model.load_state_dict(torch.load(weights_path))

    def predict(model, x1_path, x2_path):
        x1 = Image.open(x1_path)
        x2 = Image.open(x2_path)
        x1 = img_pro.resize(x1, 512, 512)
        x2 = img_pro.resize(x2, 512, 512)
        x1_ = x1
        x2_ = x2
        x1 = np.array(x1, np.float32)
        x2 = np.array(x2, np.float32)
        x1 = np.transpose(x1, [2, 0, 1]) / 255.
        x2 = np.expand_dims(x2, 0)
        x = np.concatenate([x1, x2], 0)
        x = np.expand_dims(x, 0)
        x = torch.from_numpy(x).cuda()
        model.eval()
        with torch.no_grad():
            y = model(x)[0]
        y = y.permute([1, 2, 0])
        return x1_, x2_, y

    def get_predict(img_name):
        x1_path = x1_dir + img_name + '.png'
        x2_path = x2_dir + img_name + '.png'
        x1, x2, y = predict(model, x1_path, x2_path)
        y = torch.nn.functional.softmax(y, dim=-1)
        y = torch.argmax(y, -1)
        y = Image.fromarray(np.array(colors, np.uint8)[np.array(y.cpu(), np.uint8)])
        x_y = Image.blend(x1, y, 0.3)
        x_y.save(predict_img_save_dir + img_name + '.png')
        return y, x_y

    with open(list_path) as f:
        img_names = f.readlines()
    for i in range(len(img_names)):
        img_names[i] = img_names[i].strip()

    for i in img_names:
        print(i)
        get_predict(i)



# batch_predict_('cd_b_avgg', '/media/user/PDS_data/CTX_CRATER/crater_int/crater_img/img_name_list',
#                '/media/user/PDS_data/CTX_CRATER/crater_int/crater_img/img_3c/median_3c/',
#                    '/media/user/PDS_data/CTX_CRATER/crater_int/crater_img/seg/',
#                    '/media/user/PDS_data/CTX_CRATER/crater_int/crater_img/predict/median/')