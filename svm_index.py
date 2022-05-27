import json
import torch
import numpy as np
from sklearn.externals import joblib

def get_svm(model_path):
    cls = joblib.load(model_path)

    return cls

def svm(model, features_list):
    features = np.array(features_list)
    features = features.reshape(1, -1)
    predict_label = model.predict(features)
    return predict_label[0]

if __name__ == '__main__':
    test_path = '/home/zhaomengjun/2021_binglang_paper/feature_dict/cluster_test_data.json'
    model_path = '/home/zhaomengjun/2021_binglang_paper/paper_code/logs/svm/svm.m'
    svm_model = get_svm(model_path)
    with open(test_path, 'r') as f:
        js = json.load(f)
    res_js = {'0': [0, 0, 0, 0, 0],
              '1': [0, 0, 0, 0, 0],
              '2': [0, 0, 0, 0, 0],
              '3': [0, 0, 0, 0, 0],
              '4': [0, 0, 0, 0, 0]
              }
    number = 0
    for type, vec_list in js.items():
        type = int(type)
        for per_vec in vec_list:
            number += 1
            pred = svm(svm_model, per_vec)
            res_js[str(pred)][type] += 1
            print(type, pred, number)
    ATP = 0
    print(res_js)
    for type, number_list in res_js.items():
        TP, FP, FN = 0, 0, 0
        P, R = 0, 0
        TP = res_js[type][int(type)]
        ATP += TP
        for i in range(len(res_js[type])):
            if i == int(type):
                continue
            else:
                FP += res_js[type][i]
        for pred, label_list in res_js.items():
            if int(type) == int(pred):
                continue
            else:
                FN += res_js[pred][int(type)]

        P = TP / (TP + FP)
        R = TP / (TP + FN)
        print(type, P, R)
    print(ATP / number)







