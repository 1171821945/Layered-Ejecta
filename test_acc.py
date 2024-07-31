import torch
import data_set
import assess.process_data
import file_pro.json_pro
import create_model


def acc():
    test_data_path = 'data/test_data.json'
    weights_path = 'data/train_net.pth'

    test_set_ = data_set.set('dataset/img',
                             'dataset/crater',
                             'dataset/ejecta',
                             'dataset/test_list.txt')
    train_net = create_model.create()

    test_dic = {
        'ious': {}
    }

    train_net.load_state_dict(torch.load(weights_path))
    train_net.eval()
    with torch.no_grad():
        print(assess.process_data.get_all(
            test_set_, train_net, is_log=True, test_dic=test_dic))
    file_pro.json_pro.write_json(test_dic, test_data_path)


if __name__ == 'main':
    acc()

