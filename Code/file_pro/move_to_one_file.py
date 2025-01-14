import os
import shutil


def move_to(begine_index, ori_path, aim_path):
    file_list = os.listdir(ori_path)
    file_list_ = []
    for i in file_list:
        if (i[-4:] == 'json'):
            file_list_.append(i[:-5])
    for i in range(0, len(file_list_)):
        shutil.copyfile(os.path.join(ori_path, file_list_[i] + '.json'),
                        os.path.join(aim_path, 'img{}.json'.format(i + begine_index)))
        shutil.copyfile(os.path.join(ori_path, file_list_[i] + '.png'),
                        os.path.join(aim_path, 'img{}.png'.format(i + begine_index)))
    # 返回最后的索引
    return begine_index + len(file_list_)


ori_path_list = os.listdir('all_images')
index = 0
aim_path = 'all_images_'
for i in ori_path_list:
    com_ori_path = os.path.join('all_images',i)
    index = move_to(index, com_ori_path, aim_path)



