import PIL
from PIL import Image
import numpy
with open('sum/img_name_list.txt') as f:
    lis = f.readlines()
    lis = [i.strip() for i in lis]
ori_dir = 'sum/ejecta/'
aim_dir = 'enhance/ejecta/'
suf = '.png'
for i in lis:
    img = Image.open(ori_dir + i + suf)
    img.save(aim_dir + i + suf)
    img.rotate(90, expand=1).save(aim_dir + i + '_1' + suf)
    img.rotate(180, expand=1).save(aim_dir + i + '_2' + suf)
    img.rotate(270, expand=1).save(aim_dir + i + '_3' + suf)
