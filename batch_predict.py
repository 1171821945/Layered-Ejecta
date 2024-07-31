import torch
from PIL import Image
import numpy as np
import img_pro
import create_model
path_list = 'dataset/test_list.txt'
weights_path = 'data/train_net.pth'
predict_img_save_dir = 'data/predict/'
x1_dir = 'dataset/img/'
x2_dir = 'dataset/crater/'


def batch_predict():
    colors = [(0, 0, 0), (64, 0, 0)]
    model = create_model.create()
    model.load_state_dict(torch.load(weights_path))

    def predict(model_, x1_path, x2_path):
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
        model_.eval()
        with torch.no_grad():
            y = model_(x)[0]
        y = y.permute([1, 2, 0])
        return x1_, x2_, y

    def get_predict(img_name):
        x1_path = x1_dir + img_name + '.jpg'
        x2_path = x2_dir + img_name + '.png'
        x1, x2, y = predict(model, x1_path, x2_path)
        y = torch.nn.functional.softmax(y, dim=-1)
        y = torch.argmax(y, -1)
        y = Image.fromarray(np.array(colors, np.uint8)[np.array(y.cpu(), np.uint8)])
        x_y = Image.blend(x1, y, 0.3)
        x_y.save(predict_img_save_dir + img_name + '.png')
        return y, x_y

    with open(path_list) as f:
        img_names = f.readlines()
    for i in range(len(img_names)):
        img_names[i] = img_names[i].strip()
    for i in img_names:
        print(i)
        get_predict(i)


if __name__ == 'main':
    batch_predict()