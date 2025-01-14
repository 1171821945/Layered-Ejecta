import os
import random
# folder path
imgs_dir = 'sum/img'
# Proportion of training samples
train_prop = 0.65
# Proportion of validation set
vali_prop = 0.175
# Proportion of test set
test_prop = 0.175
lis = os.listdir(imgs_dir)
all_num = len(lis)
train_num = int(all_num*train_prop)
vali_num = int(all_num*vali_prop)
test_num = all_num - train_num - vali_num
test_num = all_num - train_num
lis = [i[0:-4] for i in lis]

with open('enhance/img_name_list.txt', 'w') as f:
    for i in lis:
        f.write(i + '\n')
        f.write(i + '_1\n')
        f.write(i + '_2\n')
        f.write(i + '_3\n')

random.shuffle(lis)
with open('enhance/train_list.txt', 'w') as f:
    for i in lis[0:train_num]:
        f.write(i + '\n')
        f.write(i + '_1\n')
        f.write(i + '_2\n')
        f.write(i + '_3\n')

with open('enhance/vali_list.txt', 'w') as f:
    for i in lis[train_num:train_num + vali_num]:
        f.write(i + '\n')

with open('enhance/test_list.txt', 'w') as f:
    for i in lis[train_num + vali_num:]:
        f.write(i + '\n')