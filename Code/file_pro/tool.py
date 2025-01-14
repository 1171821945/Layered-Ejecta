import json


def get_json_dic(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        f = f.read()
        in_json_dic = json.loads(f)
    return in_json_dic


def save_json(dic, path):
    with open(path, 'w', encoding='utf-8') as f:
        st = json.dumps(dic, indent=1)
        f.write(st)