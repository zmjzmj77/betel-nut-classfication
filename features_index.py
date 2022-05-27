import json
import torch
import cv2
from model.myResNet import resnet18
from cvtorchvision import cvtransforms
from model.BetelNet import BetelNet

def get_betel(model_path):
    model = BetelNet(8,BN=True, f_flag='prelu')
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()

    return model

def predict(model, features_list):
    vec = torch.Tensor(features_list)
    vec = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(vec, 0), 0), 0)
    vec = vec.permute(0, 3, 1, 2)
    vec = vec.to(torch.device('cuda'))
    out = model(vec)
    pred = torch.max(out, 1)[1]
    pred = pred.item()
    return pred

if __name__ == '__main__':
    test_path = '/home/zhaomengjun/2021_binglang_paper/feature_dict/cluster_test_data.json'
    model_path = '/home/zhaomengjun/2021_binglang_paper/paper_code/logs/BetelNet/checkpoint/BN_prelu/201.pth'
    betelnet = get_betel(model_path)
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
            pred = predict(betelnet, per_vec)
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







