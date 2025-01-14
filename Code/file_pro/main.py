import file_pro.split_to_ind as sind
import file_pro.tool as tool


ori_dir = '../all_one_one/all_images_'
aim_dir = '../all_one_one/all_images_ind'


for i in range(0, 808):
    name = 'img'+str(i)
    dic = tool.get_json_dic(ori_dir + '/' + name + '.json')
    num = sind.get_nums(dic)
    new_dics_list = sind.get_new_dics_list(dic, name)
    sind.save_dics_list(new_dics_list, name, ori_dir, aim_dir)