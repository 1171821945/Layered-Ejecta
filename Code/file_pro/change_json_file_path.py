import json
import os

file_path = 'all/erro_img/ejecta'



def get_json_dic(file_path, file_name):
    with open(os.path.join(file_path, file_name+'.json'), 'r', encoding='utf-8') as f:
        f = f.read()
        in_json_dic = json.loads(f)
    return in_json_dic

def save_json(dic, path):
    with open(path, 'w', encoding='utf-8') as f:
        st = json.dumps(dic, indent=1)
        f.write(st)


def change_one(file_path, file_name):
    dic = get_json_dic(file_path, file_name)
    dic['imagePath'] = file_name + '.png'
    save_json(dic, os.path.join(file_path, file_name + '.json'))


file_name_list = os.listdir(file_path)
file_name_list_ = []
for i in file_name_list:
    if (i[-4:] == 'json'):
        file_name_list_.append(i[:-5])
for i in file_name_list_:
    change_one(file_path, i)
