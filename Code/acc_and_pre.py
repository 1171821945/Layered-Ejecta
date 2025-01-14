import vali_acc
import test_acc
import batch_predict
import os



path2 = 'best_loss.pth'
net_name = 'emhseu_unet'
#
dir1 = 'data/{}/test_data/{}/'.format(net_name, path2[:-4])

# batch_predict.batch_predict(net_name, dir1 + path2, 'data/{}/test_data/{}/'.format(net_name, path2[:-4]))
for path2 in ['best_iou.pth', 'best_f1.pth', 'best_loss.pth']:
    dir1 = 'data/{}/test_data/{}/'.format(net_name, path2[:-4])
    test_acc.acc(net_name, dir1 + path2, 'data/{}/test_data/'.format(net_name) + path2[:-4] + '/test_data.json')





# for d in ['best_iou', 'best_f1', 'best_loss']:
#     if not os.path.isdir('data/{}/test_data/{}'.format(net_name, d)):
#         os.makedirs('data/{}/test_data/{}'.format(net_name, d))





#test_acc.acc()
#vali_acc.acc(net_name)
#vali_acc.acc(net_name)
#batch_predict.batch_predict(net_name, is_best)
