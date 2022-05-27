import readjson
import os
import shutil
from random import sample

def copy_file(key, sample_index, label):
    new_base_path = os.path.join('/home/zhaomengjun/2021_binglang_paper/dataset_end', label, key)
    if not os.path.exists(new_base_path):
        os.makedirs(new_base_path)
    img_index = 0
    if label == 'con':
        for per_index in sample_index:
            new_path = os.path.join(new_base_path, '{}.bmp'.format(img_index))
            shutil.copyfile(per_index, new_path)
            img_index += 1
            print(img_index)
    else:
        for per_index in sample_index:
            per_index = per_index.replace('con', 'cut')
            new_path = os.path.join(new_base_path, '{}.bmp'.format(img_index))
            shutil.copyfile(per_index, new_path)
            img_index += 1
            print(img_index)

if __name__ == '__main__':
    json_path = '/home/zhaomengjun/2021_binglang_paper/data_path/path.json'
    with open(json_path, 'r') as f:
        js = json.load(f)
    for key, path_list in js.items():
        print(key)
        if key != '6':
            sample_index = sample(path_list, 1100)
        else:
            sample_index = path_list

        copy_file(key, sample_index, 'con')
        copy_file(key, sample_index, 'cut')