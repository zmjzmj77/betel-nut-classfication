import json
from sklearn import svm
import os
import numpy as np
from sklearn.metrics import accuracy_score
import pickle
from sklearn.externals import joblib

def read_json(json_path):
    with open(json_path, 'r') as f:
        js = json.load(f)

    data_list = []
    label_list = []
    for label, value in js.items():
        if label not in ['0', '1', '2', '3', '4']:
            continue
        label = int(label)
        for per in value:
            data_list.append(per)
            label_list.append(label)
    print(len(data_list), len(label_list))

    return data_list, label_list

def save_model(model, save_path):
    with open(os.path.join(save_path, 'svm.txt'), 'w') as f:
        f.write(model)

if __name__ == '__main__':
    train_path = '/home/zhaomengjun/2021_binglang_paper/feature_dict/cluster_train_data.json'
    test_path = '/home/zhaomengjun/2021_binglang_paper/feature_dict/cluster_test_data.json'
    save_path = '/home/zhaomengjun/2021_binglang_paper/paper_code/logs/svm'

    train_data, train_label = read_json(train_path)
    test_data, test_label = read_json(test_path)

    classfier = svm.SVC(C=1, kernel='rbf', gamma=1, decision_function_shape='ovo')
    classfier.fit(train_data, train_label)

    print('train end')
    # model = pickle.dumps(classfier)
    # save_model(model, save_path)
    joblib.dump(classfier, os.path.join(save_path, 'svm.m'))

    cls = joblib.load(os.path.join(save_path, 'svm.m'))
    predict_label = cls.predict(train_data)
    accuracy_rate = accuracy_score(train_label, predict_label)
    print('train accuracy rate is {}'.format(accuracy_rate))

    predict_label = cls.predict(test_data)
    accuracy_rate = accuracy_score(test_label, predict_label)
    print('test accuracy rate is {}'.format(accuracy_rate))
