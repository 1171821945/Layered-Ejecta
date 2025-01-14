import json
import os
import copy
import shutil
import file_pro.tool as tool


def get_nums(dic):
    return len(dic['shapes'])//2


def get_new_dics_list(dic, name):
    num = get_nums(dic)
    new_dics_list = []
    for i in range(0, num):
        new_dic = copy.deepcopy(dic)
        shape_list = copy.deepcopy(new_dic['shapes'])
        new_dic['shapes'] = shape_list[i*2:i*2+2]
        new_dic['imagePath'] = name+'_'+str(i)+'.png'
        new_dics_list.append(new_dic)
    return new_dics_list


def save_dics_list(dics_list, name, ori_dir, aim_dir):
    num = len(dics_list)
    for i in range(0, num):
        shutil.copyfile(os.path.join(ori_dir, name+'.png'), os.path.join(aim_dir, name+'_{}'.format(i)+'.png'))
        tool.save_json(dics_list[i], os.path.join(aim_dir, name+'_{}'.format(i)+'.json'))

