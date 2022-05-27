import os
import numpy as np
import readjson

def get_feature(read_base_path):
    feature_list = []
    for root, dirs, files in os.walk(read_base_path):
        for file in files:
            index = 0
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                with open(json_path, 'r') as f:
                    js = json.load(f)
                for _, dict in js.items():
                    sub_list = [val for _, val in dict.items()]
                    # sub_list = []
                    # n = 0
                    # for _, val in dict.items():
                    #     n += 1
                    #     if n < 9:
                    #         continue
                    #     sub_list.append(val)
                    feature_list.append(sub_list)
                    index += 1
                print(index)
    print(len(feature_list))
    features = np.array(feature_list)
    return features

def select_k(features_arr):
    n_samples, n_features = features_arr.shape
    min_v = np.array([np.min(features_arr[:, i]) for i in range(0, n_features)])
    max_v = np.array([np.max(features_arr[:, i]) for i in range(0, n_features)])
    mean_v = np.array([np.mean(features_arr[:, i]) for i in range(0, n_features)])
    std_v = np.array([np.std(features_arr[:, i]) for i in range(0, n_features)])
    norm_arr = (features_arr - mean_v) / std_v
    cov_matrix = np.dot(np.transpose(norm_arr), norm_arr)
    U, S, V = np.linalg.svd(cov_matrix)
    print(S, len(S))

    for i in range(1, len(S)+1):
        print(np.sum(S[:i]) / np.sum(S), i)

def pca(features_arr, k, pca_path):
    n_samples, n_features = features_arr.shape
    print(n_features)
    min_v = np.array([np.min(features_arr[:, i]) for i in range(0, n_features)])
    max_v = np.array([np.max(features_arr[:, i]) for i in range(0, n_features)])
    mean_v = np.array([np.mean(features_arr[:, i]) for i in range(0, n_features)])
    std_v = np.array([np.std(features_arr[:, i]) for i in range(0, n_features)])
    norm_arr = (features_arr - mean_v) / std_v
    cov_matrix = np.dot(np.transpose(norm_arr), norm_arr)
    eig_val, eig_vec = np.linalg.eig(cov_matrix)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]
    eig_pairs.sort(reverse=True)
    feature = [list(ele[1]) for ele in eig_pairs[:k]]

    js = {'mean': list(mean_v),
          'std': list(std_v),
          'min': list(min_v),
          'max': list(max_v),
          'feature': feature
          }

    with open(os.path.join(pca_path, 'texture_pca_stand.json'), 'w') as f:
        json.dump(js, f, indent=4)

if __name__ == '__main__':
    read_base_path = '/home/zhaomengjun/2021_binglang_paper/feature_dict/cluster_con_feature'
    pca_path = '/home/zhaomengjun/2021_binglang_paper/feature_dict/pca_parameter'
    features_arr = get_feature(read_base_path)
    select_k(features_arr)

    # k = 4
    # pca(features_arr, k, pca_path)