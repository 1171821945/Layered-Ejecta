import json


def read_json(path):
    with open(path) as f:
        dic = json.load(f)
    return dic


def write_json(dic, path):
    with open(path, 'w') as f:
        f.write(json.dumps(dic, indent=1))
