import readjson
import os
import numpy as np
from sklearn.cluster import KMeans

def read_js(read_path):
    name_list = []
    feature_list = []
    with open(read_path, 'r') as f:
        js = json.load(f)
    for name, val in js.items():
        name_list.append(name)
        feature_list.append(val)
    return name_list, feature_list

def kmeans(feature_list):
    feature_arr = np.array(feature_list)
    pred = KMeans(n_clusters=5).fit_predict(feature_arr)
    pred = list(pred)

    return pred

def save_feature(all_dict,  name_list, feature_list, type_list, pred, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    js = {}
    name_js = {}
    for i in range(len(pred)):
        type = str(type_list[pred[i]])
        if type not in js.keys():
            js[type] = []
            name_js[type] = []
        js[type].append(feature_list[i])
        name_js[type].append(name_list[i])

    for type in js.keys():
        save_path1 = os.path.join(save_path, type)
        if not os.path.exists(save_path1):
            os.makedirs(save_path1)
        feature_dict = {}
        for i in range(len(js[type])):
            feature_dict[name_js[type][i]] = all_dict[name_js[type][i]]
        save_path2 = os.path.join(save_path1, 'features_dict.json')
        with open(save_path2, 'w') as f:
            json.dump(feature_dict, f, indent=4)
        print(type, len(feature_dict.keys()))

def get_feature(read_base_path):
    feature_dict = {}
    for root, dirs, files in os.walk(read_base_path):
        for file in files:
            if file.endswith('.json'):
                type = os.path.basename(root)
                json_path = os.path.join(root, file)
                with open(json_path, 'r') as f:
                    js = json.load(f)
                for name, dict in js.items():
                    label = '{}_{}'.format(type, name)
                    feature_dict[label] = dict
    return feature_dict

if __name__ == '__main__':
    read_path = '/home/zhaomengjun/2021_binglang_paper/feature_dict/con_feature/all_feature.json'
    read_base_path = '/home/zhaomengjun/2021_binglang_paper/feature_dict/con_feature'
    all_dict = get_feature(read_base_path)
    print(len(all_dict.keys()))
    name_list, feature_list = read_js(read_path)
    pred = kmeans(feature_list)

    index_list = [[], [], [], [], []]

    for i in range(len(pred)):
        index = int(pred[i])
        index_list[index].append(int(name_list[i].split('_')[0]))
    type_list = []
    for sub_list in index_list:
        counts = np.bincount(sub_list)
        label = np.argmax(counts)
        print(len(sub_list), label)
        type_list.append(label)

    save_path = '/home/zhaomengjun/2021_binglang_paper/feature_dict/cluster_con_feature'
    save_feature(all_dict, name_list, feature_list, type_list, pred, save_path)
