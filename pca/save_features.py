import os
import glob
import readjson
import random

import numpy as np

def get_pca_features(json_path, mean, std, min_v, max_v, pca_vec):
    with open(json_path, 'r') as f:
        js = json.load(f)
    sample_features = []
    per_name = []

    for name, feature_dict in js.items():
        per_features = [val for _, val in feature_dict.items()]
        per_name.append(name)
        sample_features.append(per_features)
    print(len(sample_features), len(per_name))
    sample_features = np.array(sample_features)
    norm_features = (sample_features - mean) / std    #标准化后的特征

    print(norm_features.shape)
    shape_features = list(norm_features[:,:8])
    texture_features = list(norm_features[:,8:])

    pca_vec = np.array(pca_vec)
    print(pca_vec.shape)
    result = np.dot(norm_features, np.transpose(pca_vec))
    result = list(result)

    return result, per_name, shape_features, texture_features

def get_pca(pca_path, k):
    with open(pca_path, 'r') as f:
        js = json.load(f)
    mean = js['mean']
    std = js['std']
    min_v = js['min']
    max_v = js['max']
    pca_vec = js['feature'][:k]

    return mean, std, min_v, max_v, pca_vec

def get_sample(result, name, shape_features, texture_features):
    length = len(result)
    train_len = length * 8 // 10
    all_index = [i for i in range(0, length)]
    train_index = random.sample(all_index, train_len)
    test_index = list(set(all_index).difference(set(train_index)))

    sub_train_sample = [list(result[i]) for i in train_index]
    sub_test_sample = [list(result[i]) for i in test_index]

    train_name = [name[i] for i in train_index]
    test_name = [name[i] for i in test_index]

    sub_train_shape_sample = [list(shape_features[i]) for i in train_index]
    sub_test_shape_sample = [list(shape_features[i]) for i in test_index]

    sub_train_texture_sample = [list(texture_features[i]) for i in train_index]
    sub_test_texture_sample = [list(texture_features[i]) for i in test_index]

    return  sub_train_sample, sub_test_sample, train_name, test_name, sub_train_shape_sample, sub_test_shape_sample,\
            sub_train_texture_sample, sub_test_texture_sample

def save_json(save_path, sample, flag):
    with open(os.path.join(save_path, 'cluster_{}_data.json'.format(flag)), 'w') as f:
        json.dump(sample, f , indent=4)

def get_mean_std(all_result):
    all_result = np.array(all_result)
    all_mean, all_std, all_min, all_max = [], [], [], []
    for i in range(0, all_result.shape[1]):
        all_mean.append(np.mean(all_result[:, i]))
        all_std.append(np.std(all_result[:, i]))
        all_min.append(np.min(all_result[:, i]))
        all_max.append(np.max(all_result[:, i]))

    print(len(all_mean), len(all_std))
    return all_mean, all_std, all_min, all_max

if __name__ == '__main__':
    pca_path = '/home/zhaomengjun/2021_binglang_paper/feature_dict/pca_parameter/pca_stand.json'
    sample_path = '/home/zhaomengjun/2021_binglang_paper/feature_dict/cluster_con_feature'
    save_path = '/home/zhaomengjun/2021_binglang_paper/feature_dict'
    save_path1 = '/home/zhaomengjun/2021_binglang_paper/dataset_end/224_con'

    all_result = []
    train_sample, test_sample = {}, {}
    train_name, test_name = {}, {}
    train_shape_sample, test_shape_sample = {}, {}
    train_texture_sample, test_texture_sample = {}, {}
    mean, std, min_v, max_v, pca_vec = get_pca(pca_path, 8)
    for per_type in glob.glob('{}/*'.format(sample_path)):
        type = str(int(os.path.basename(per_type)) - 4)

        json_path = os.path.join(per_type, 'features_dict.json')
        result, name, shape_features, texture_features = get_pca_features(json_path, mean, std, min_v, max_v, pca_vec)
        print(len(result))

        sub_train_sample, sub_test_sample, sub_train_name, sub_test_name,\
            sub_train_shape_sample, sub_test_shape_sample,sub_train_texture_sample, sub_test_texture_sample\
                                                            = get_sample(result, name, shape_features, texture_features)
        print(len(sub_train_sample), len(sub_train_name), len(sub_train_shape_sample), len(sub_train_texture_sample))

        train_sample[type] = sub_train_sample
        test_sample[type] = sub_test_sample

        for i in range(len(sub_train_name)):
            sub_train_name[i] = '{}/{}/{}_{}'.format(save_path1, os.path.basename(per_type), os.path.basename(per_type), sub_train_name[i])
        train_name[os.path.basename(per_type)] = sub_train_name
        for i in range(len(sub_test_name)):
            sub_test_name[i] = '{}/{}/{}_{}'.format(save_path1, os.path.basename(per_type), os.path.basename(per_type), sub_test_name[i])
        test_name[os.path.basename(per_type)] = sub_test_name

        train_shape_sample[type] = sub_train_shape_sample
        test_shape_sample[type] = sub_test_shape_sample

        train_texture_sample[type] = sub_train_texture_sample
        test_texture_sample[type] = sub_test_texture_sample

        for i in result:
            all_result.append(i)
    print(len(all_result))
    all_mean, all_std, all_min, all_max = get_mean_std(all_result)
    # train_sample['mean'] = all_mean
    # train_sample['std'] = all_std
    # train_sample['min'] = all_min
    # train_sample['max'] = all_max
    #
    # test_sample['mean'] = all_mean
    # test_sample['std'] = all_std
    # test_sample['min'] = all_min
    # test_sample['max'] = all_max


    save_json(save_path, train_sample, 'train')
    save_json(save_path, test_sample, 'test')

    with open(os.path.join(save_path1, 'train.json'), 'w') as f:
        json.dump(train_name, f, indent=4)
    with open(os.path.join(save_path1, 'test.json'), 'w') as f:
        json.dump(test_name, f, indent=4)


    save_json(save_path, train_shape_sample, 'shape_train')
    save_json(save_path, test_shape_sample, 'shape_test')

    save_json(save_path, train_texture_sample, 'texture_train')
    save_json(save_path, test_texture_sample, 'texture_test')

