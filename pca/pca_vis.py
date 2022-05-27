import os
import readjson
import glob
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import random

def get_pca_features(json_path, mean, std, min_v, max_v, pca_vec, k):
    with open(json_path, 'r') as f:
        js = json.load(f)
    sample_features = []
    for _, feature_dict in js.items():
        per_features = [val for _, val in feature_dict.items()]
        sample_features.append(per_features)
    sample_features = np.array(sample_features)
    norm_features = (sample_features - mean) / std
    pca_vec = np.array(pca_vec)
    print(pca_vec.shape)
    result = np.dot(norm_features, np.transpose(pca_vec))
    if k == 3:
        list1, list2, list3 = result[:, 0],  result[:, 1], result[:, 2]

        return list1, list2, list3
    else:
        list1, list2 = result[:, 0], result[:, 1]

        return list1, list2

def get_vec(pca_path, k):
    with open(pca_path, 'r') as f:
        js = json.load(f)
    mean = js['mean']
    std = js['std']
    min_v = js['min']
    max_v = js['max']
    pca_vec = js['feature'][:k]

    return mean, std, min_v, max_v, pca_vec

def draw_plt(all_list, save_path, k):
    if k == 3:
        matplotlib.rcParams['font.size'] = 20
        fig = plt.figure(figsize=(14,14))
        ax = plt.axes(projection='3d')
        for per_list in all_list:
            ax.scatter(per_list[0], per_list[1], per_list[2],label= per_list[3])
        ax.view_init(elev=20, azim=30)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.legend()
        ax.grid(False)
        # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        plt.savefig(os.path.join(save_path, 'cluster_pca_stand_{}.jpg'.format(k)))
        plt.show()
        plt.close()
    else:
        plt.figure(figsize=(14, 14))
        for per_list in all_list:
            plt.scatter(per_list[0], per_list[1], label=per_list[2])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    read_path = r'E:\dataset_end\cluster_con_feature'
    pca_path = r'E:\dataset_end\data_vis\cluster_pca_stand.json'
    save_path = r'E:\dataset_end\data_vis'
    all_list_3 = []
    all_list_2 = []
    for type_path in glob.glob('{}/*'.format(read_path)):
        type = os.path.basename(type_path)
        json_path = os.path.join(type_path, 'features_dict.json')
        mean, std, min_v, max_v, pca_vec = get_vec(pca_path, 3)
        list1, list2, list3 = get_pca_features(json_path, mean, std, min_v, max_v, pca_vec, 3)
        all_list_3.append([list1, list2, list3, type])

        # mean1, pca_vec1 = get_vec(pca_path, 2)
        # list1, list2= get_pca_features(json_path, mean1, pca_vec1, 2)
        # all_list_2.append([list1, list2, type])

    draw_plt(all_list_3, save_path, 3)
    # draw_plt(all_list_2, save_path, 2)