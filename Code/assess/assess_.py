import numpy as np
import os
import cv2
import file_pro.json_pro as jp
import torch


# 根据阈值判断每个像素点属于哪种类型
def get_y2(y, thresh, ori = False):
    if not ori:
        y2 = np.where(y > thresh, 1, 0)
        y2 = y2.astype('uint8')
    else:
        y2 = np.array(torch.sigmoid(torch.from_numpy(y)))
        y2 = np.where(y2 > thresh, 1, 0)
        y2 = y2.astype('uint8')
    return y2


# 获取iou
def get_iou(ac, pre):
    i = (np.sum(ac*pre))
    u = (np.sum(ac + pre) - np.sum(ac*pre))
    iou = i/u
    return iou


# 混淆矩阵
# TP FP
# FN TN
def get_conmat(ac, pre):
    TP = np.sum(ac*pre)
    FP = np.sum(pre - ac*pre)
    FN = np.sum(ac - ac*pre)
    TN = np.sum((1-ac)*(1-pre))
    sum = TP + FP + FN + TN
    return TP/sum, FP/sum, FN/sum, TN/sum


# TPR FPR
def get_tf(ac, pre):
    TPR = np.sum(ac*pre)/np.sum(ac)
    FPR = np.sum(pre - ac*pre)/np.sum(1-ac)
    return TPR, FPR


backbone = 'vgg'
if backbone == 'vgg':
    np_dir = '../batch_predict/vgg/pre_y/'
    png_dir = '../enhance/ejecta_resize/'
    roc_point_path = '../data/data_vgg/roc.json'
else:
    np_dir = '../batch_predict/resnet/pre_y/'
    png_dir = '../enhance/ejecta_resize/'
    roc_point_path = '../data/data_resnet/roc.json'

lis = os.listdir(np_dir)
lis = [i[0:-4] for i in lis]
test_nums = 904


def get_miou():
    ious = 0
    for i in range(0, test_nums):
        img = cv2.imread(png_dir + lis[i] + '.png', -1)
        y = np.load(np_dir + lis[i] + '.npy')[:, :, 1]
        y = get_y2(y, 0.5)
        ious += get_iou(img, y)
    return ious / test_nums


# 对结果取以10为底的对数值
def get_mtf(thresh):
    sum_TP = 0
    sum_FP = 0
    for i in range(0, test_nums):
        img = cv2.imread(png_dir + lis[i] + '.png', -1)
        y = np.load(np_dir + lis[i] + '.npy')[:, :, 1]
#        y = np.log10(y)
        y = get_y2(y, thresh)
        pre, recall = get_tf(img, y)
        sum_TP += pre
        sum_FP += recall
    return sum_TP/test_nums, sum_FP/test_nums


def get_roc_point():
    points = []
    threshes = [i*0.05 for i in range(1, 20)]
    for i in threshes:
        mpre, mrecall = get_mtf(i)
        points.append([i, mpre, mrecall])
        print([i, mpre, mrecall])
    return points


img = cv2.imread(png_dir + lis[4] + '.png', -1)
y = np.load(np_dir + lis[4] + '.npy')[:, :, 1]


def get_roc():
    # threshes = [0, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2,
    #             0.1, 0.5, 0.9, 1 - 1e-2, 1 - 5e-3, 1 - 1e-3,
    #             1 - 5e-4, 1 - 1e-4, 1 - 5e-5, 1 - 1e-5, 1 - 1e-6, 1]

    threshes = list(range(-25, 0))
    threshes_ = [-1e-1, -1e-2, -1e-3, -1e-4, -1e-5, -1e-6, -1e-7, -1e-8, -1e-9, -1e-10, 0]
    threshes = threshes + threshes_
    points = []
    for i in range(0, len(threshes)):
        th = threshes[i]
        points.append(get_mtf(threshes[i]))
        print(th)
        print(points)
    jp.write_json({'threshes': threshes, 'T_F_points': points}, roc_point_path)


def show(i, thresh):
    y = np.load(np_dir + lis[i] + '.npy')[:, :, 1]
    y = get_y2(y)
