import json
import torch
import cv2
from model.myResNet import resnet18
from cvtorchvision import cvtransforms

def create_model(resnet, model_path):
    model = resnet()
    if model_path is not None:
        checkpoint = torch.load(model_path)
        model_dict = model.state_dict()
        checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
    model.cuda()
    model.eval()
    return model

def transforms():
    transform = cvtransforms.Compose([
        cvtransforms.ToTensor(),
        cvtransforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform

def resnet(or_img, model):
    # crop_img = or_img[12:1012, 140:1140]
    # res_img = cv2.resize(crop_img, (224, 224))
    img = cv2.cvtColor(or_img, cv2.COLOR_BGR2RGB)
    transform = transforms()
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    img = img.to(torch.device('cuda'))
    out = model(img)
    # print(out)
    pred = torch.max(out, 1)[1]
    pred = pred.item()

    return pred
if __name__ == '__main__':
    test_path = '/home/zhaomengjun/2021_binglang_paper/dataset_end/224_con/test.json'
    model_path = '/home/zhaomengjun/2021_binglang_paper/paper_code/logs/resnest18/checkpoint/618.pth'
    model = create_model(resnet18, model_path)
    with open(test_path, 'r') as f:
        js = json.load(f)
    res_js = {'0':[0,0,0,0,0],
              '1':[0,0,0,0,0],
              '2':[0,0,0,0,0],
              '3':[0,0,0,0,0],
              '4':[0,0,0,0,0]
              }
    number = 0
    for type, name_list in js.items():
        type = int(type) - 4
        for per_path in name_list:
            number += 1
            img = cv2.imread(per_path)
            pred = resnet(img, model)
            res_js[str(pred)][type] += 1
            print(type, per_path, pred, number)
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
    print(ATP/number)







