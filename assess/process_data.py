import numpy as np
import torch



def getCE(inp, tag, is_imp_loss):
    h, w, c = inp.size()
    ht, wt = tag.size()
    inp = inp.view(-1, c)
    tag = tag.view(-1).cuda()
    loss = float(torch.nn.CrossEntropyLoss()(inp, tag))
    if is_imp_loss:
        loss = float(loss/torch.sum(tag))
    return float(loss)


def get_loss(dataset, model):
    loss = 0
    for x, y in dataset:
        x = np.expand_dims(x, 0)
        x = torch.from_numpy(x).cuda()
        y_pre = model(x)[0]
        y_pre = y_pre.permute([1, 2, 0]).contiguous()
        loss += getCE(y_pre, y)
    return loss/len(dataset)


def get_iou(ac, pre):
    i = (torch.sum(ac*pre))
    u = (torch.sum(ac + pre) - torch.sum(ac*pre))
    iou = float(i/u)
    return iou




def get_miou(dataset, model):
    iou = 0
    for x, y in dataset:
        x = np.expand_dims(x, 0)
        x = torch.from_numpy(x).cuda()
        y_pre = model(x)[0]
        y_pre = y_pre.permute([1, 2, 0])
        y_pre = torch.argmax(y_pre, -1)
        one_iou = get_iou(y.cuda(), y_pre)
#        print(one_iou)
        iou += one_iou
    return iou/len(dataset)


def get_conmat(ac, pre):
    TP = float(torch.sum(ac*pre))
    FP = float(torch.sum(pre - ac*pre))
    FN = float(torch.sum(ac - ac*pre))
    TN = float(torch.sum((1-ac)*(1-pre)))
    sum = TP + FP + FN + TN
    return TP/sum, FP/sum, FN/sum, TN/sum


def get_mean_conmat(dataset, model):
    sum_con_mat = [0, 0, 0, 0]
    for x, y in dataset:
        x = np.expand_dims(x, 0)
        x = torch.from_numpy(x).cuda()
        y_pre = model(x)[0]
        y_pre = y_pre.permute([1, 2, 0])
        y_pre = torch.argmax(y_pre, -1)
        con_mat = get_conmat(y.cuda(), y_pre)
        sum_con_mat[0] += con_mat[0]
        sum_con_mat[1] += con_mat[1]
        sum_con_mat[2] += con_mat[2]
        sum_con_mat[3] += con_mat[3]
    return sum_con_mat[0] / len(dataset), sum_con_mat[1] / len(dataset), sum_con_mat[2] / len(dataset), sum_con_mat[3] / len(dataset)


def get_all(dataset, model, is_imp_loss=True, is_log=False, test_dic=None):
    sum_con_mat = [0, 0, 0, 0]
    loss = 0
    iou = 0
    precisions = 0
    recalls = 0
    f1score = 0
    f2score = 0
    for x, y, name in dataset:
        x = np.expand_dims(x, 0)
        x = torch.from_numpy(x).cuda()
        y_pre = model(x)[0]
        y_pre = y_pre.permute([1, 2, 0])
        loss += getCE(y_pre, y, is_imp_loss)
        y_pre = torch.argmax(y_pre, -1)
        con_mat = get_conmat(y.cuda(), y_pre)
        precision = con_mat[0]/((con_mat[0] + con_mat[1])+1e-8)
        recall = con_mat[0]/((con_mat[0] + con_mat[2])+1e-8)
        precisions += precision
        recalls += recall
        sum_con_mat[0] += con_mat[0]
        sum_con_mat[1] += con_mat[1]
        sum_con_mat[2] += con_mat[2]
        sum_con_mat[3] += con_mat[3]
        one_iou = get_iou(y.cuda(), y_pre)
        iou += one_iou
        f1score += 2*precision*recall/(precision + recall + 1e-8)
        f2score += 5*precision*recall/(4*precision + recall + 1e-8)
        if is_log:
            test_dic['ious'][name] = one_iou

    if is_log:
        test_dic['precision'] = precisions/len(dataset)
        test_dic['recall'] = recalls/len(dataset)
        test_dic['con_mat'] = [sum_con_mat[0] / len(dataset), sum_con_mat[1] / len(dataset), sum_con_mat[2] / len(dataset), sum_con_mat[3] / len(dataset)]
        test_dic['iou'] = iou/len(dataset)
        test_dic['f1score'] = f1score/(len(dataset))
        test_dic['f2score'] = f2score/(len(dataset))


    return loss/len(dataset), \
           [sum_con_mat[0] / len(dataset), sum_con_mat[1] / len(dataset), sum_con_mat[2] / len(dataset), sum_con_mat[3] / len(dataset)], \
           iou/len(dataset), \
           precisions/len(dataset), \
           recalls/len(dataset),\
           f1score/(len(dataset)), \
           f2score/(len(dataset))



