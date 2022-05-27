import readjson
import os
import glob

def find_json(con_base_path):
    json_list = []
    for root, dirs, files in os.walk(con_base_path):
        for file in files:
            if file == 'path_dict.json':
                json_list.append(os.path.join(root, file))

    return json_list

def get_dict(json_list):
    dict = {}
    for per_json in json_list:
        label = per_json.split('/')[-2]
        with open(per_json, 'r') as f:
            js = json.load(f)
            dict[label] = js['correct_p_8']

    return dict

def get_and(con_dict, cut_dict):
    end_dict = {}
    for key in con_dict.keys():
        con_list = con_dict[key]
        cut_list = cut_dict[key]
        for i in range(0, len(cut_list)):
            cut_list[i] = cut_list[i].replace('cut', 'con')
        and_list = list(set(con_list).intersection(set(cut_list)))
        end_dict[key] = and_list

    return end_dict

if __name__ == '__main__':
    con_base_path = '/home/zhaomengjun/2021_binglang_paper/data_path/con'
    cut_base_path = '/home/zhaomengjun/2021_binglang_paper/data_path/cut'
    save_path = '/home/zhaomengjun/2021_binglang_paper/data_path'
    con_json_list = find_json(con_base_path)
    cut_json_list = find_json(cut_base_path)

    con_label_path_dict = get_dict(con_json_list)
    cut_label_path_dict = get_dict(cut_json_list)

    end_dict = get_and(con_label_path_dict, cut_label_path_dict)

    with open(os.path.join(save_path, 'path.json'), 'w') as f:
        json.dump(end_dict, f, indent=4)

    for key in end_dict.keys():
        print(key, len(end_dict[key]))
