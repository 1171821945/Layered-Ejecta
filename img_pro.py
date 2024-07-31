import numpy as np
from PIL import Image


def resize(img, width=512, height=512):
    img_w = img.width
    img_h = img.height
    mode = 'RGB'
    if(len(np.array(img).shape) == 2):
        mode = 'L'
        img = Image.fromarray(np.array(img), 'L')
    in_w = 0
    in_h = 0
    w_scale = img_w/width
    h_scale = img_h/height
    if(w_scale>=h_scale):
        in_w = width
        in_h = int(img_h/img_w*width+0.5)
        in_img = img.resize([in_w, in_h])
    else:
        in_h = height
        in_w = int(img_w/img_h*height)
        in_img = img.resize([in_w, in_h])
    back_img = Image.new(mode, (width, height))
    back_img.paste(in_img, ((width-in_w)//2, (height-in_h)//2))
    return back_img