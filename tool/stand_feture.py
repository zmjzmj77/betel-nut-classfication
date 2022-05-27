import readjson
import numpy as np
import os

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
                    sub_list = [val for _, val in dict.items()]
                    label = '{}_{}'.format(type, name)
                    feature_dict[label] = sub_list
    return feature_dict

def stand_func(feature_dict):
    feature_list = []
    for _, val in feature_dict.items():
        feature_list.append(val)
    print(len(feature_list))
    features_arr = np.array(feature_list)

    n_samples, n_features = features_arr.shape
    mean_v = list([np.mean(features_arr[:, i]) for i in range(0, n_features)])
    std_v = list([np.std(features_arr[:, i]) for i in range(0, n_features)])
    print(len(mean_v), std_v)

    for name, val in feature_dict.items():
        for i in range(len(val)):
            val[i] = (val[i] - mean_v[i]) / std_v[i]
        feature_dict[name] = val

    return feature_dict

def save_js(stand_feature, save_path):
    save_path = os.path.join(save_path, 'all_feature.json')
    with open(save_path, 'w') as f:
        json.dump(stand_feature, f, indent=4)

if __name__ == '__main__':
    read_base_path = r'E:\dataset_end\con_feature'
    feature_dict = get_feature(read_base_path)
    stand_feature = stand_func(feature_dict)
    save_js(stand_feature, read_base_path)