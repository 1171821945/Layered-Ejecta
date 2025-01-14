import json
import os
import copy
import shutil


def get_json_dic(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        f = f.read()
        in_json_dic = json.loads(f)
    return in_json_dic

def label_dic_crater(label_dic):
    shapes = label_dic['shapes']
    shapes_of_crater = []
    for i in shapes:
        if(i['label'] == 'crater'):
            shapes_of_crater.append(i)
    label_dic = copy.deepcopy(label_dic)
    label_dic['shapes'] = shapes_of_crater
    return label_dic

def label_dic_ejecta(label_dic):
    shapes = label_dic['shapes']
    shapes_of_ejecta = []
    for i in shapes:
        if(i['label'] == 'ejecta'):
            shapes_of_ejecta.append(i)
    label_dic = copy.deepcopy(label_dic)
    label_dic['shapes'] = shapes_of_ejecta
    return label_dic

def save_json(dic, path):
    with open(path, 'w', encoding='utf-8') as f:
        st = json.dumps(dic, indent=1)
        f.write(st)
ori_dir = 'all_one_one/all_images_ind'
crater_aim_dir = 'all_one_one/all_images_crater_ind'
ejecta_aim_dir = 'all_one_one/all_images_ejecta_ind'
lis = os.listdir(ori_dir)
lis2 = []
for i in lis:
    if (i[-4:] == 'json'):
        lis2.append(i)
for i in lis2:
    name = i[0:-4]
    img_name = name+'png'
    shutil.copyfile('{}/{}'.format(ori_dir,img_name), '{}/{}'.format(crater_aim_dir, img_name))
    json_dic = get_json_dic(os.path.join(ori_dir, i))
    c_j = label_dic_crater(json_dic)
    save_json(c_j, '{}/{}'.format(crater_aim_dir, i))


