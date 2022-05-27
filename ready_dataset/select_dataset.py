import sys
sys.path.insert(0, './save_model')
import numpy as np
import glob
import os
import torch
import cv2
from cvtorchvision import cvtransforms
import readjson
def transform():
    transforms = cvtransforms.Compose([
        cvtransforms.ToTensor(),
        cvtransforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ])
    return transforms

def pre_process(img):
    crop_img = img[0:1024, 128:1152]
    mask = np.zeros(crop_img.shape[:2], dtype='uint8')
    cv2.rectangle(mask, (322, 0), (702, 1024), 255, -1)
    black_img = cv2.bitwise_and(crop_img, crop_img, mask = mask)
    img = cv2.resize(black_img, (224,224))

    return img

def inference(img_path, model_path, transforms):
    img = cv2.imread(img_path)

    img = pre_process(img)

    img_trans = transforms(img)
    img_tensor = torch.unsqueeze(img_trans, dim=0)
    img_tensor.cuda()

    model = torch.load(model_path, map_location="cuda:0")
    # model.cuda()
    model.eval()

    out = model(img_tensor)
    index = torch.max(out, 1)[1]
    out_p = torch.nn.functional.softmax(out, 1)
    max_p = torch.max(out_p, 1)[0].float()
    return index, max_p

def get_acc(path_list, model_path, transforms, label):
    num = 0
    all = 0
    path = {'out_path':[],
            'wrong_path' : [],
            'correct_p_8':[],
            'correct_p_5':[],
            'correct_p_0':[]
            }
    pre_index = 9
    pre_out_p = 0
    len_08, len_05, len_0 = 0, 0, 0
    for img_path in path_list:
        try:
            index, out_p = inference(img_path, model_path, transforms)
        except:
            continue
        print(index, out_p)
        if index == pre_index and out_p == pre_out_p:
            path['out_path'].append(img_path)
            continue
        pre_index = index
        pre_out_p = out_p

        if index == label:
            num += 1
            if out_p > 0.8:
                path['correct_p_8'].append(img_path)
                len_08 += 1
            elif out_p > 0.5:
                path['correct_p_5'].append(img_path)
                len_05 += 1
            else:
                path['correct_p_0'].append(img_path)
                len_0 += 1
        elif index < 4:
            path['out_path'].append(img_path)
        else:
            path['wrong_path'].append(img_path)

        all += 1
    acc = num / all
    path['sub_len'] = [len_08, len_05, len_0]
    return path, acc

def find_path(img_path):
    path_list = []
    for per_channel in glob.glob('{}/*'.format(img_path)):
        bmp_name = []
        for per_bmp in os.listdir(per_channel):
            bmp_name.append(int(per_bmp[:-4]))
        bmp_name = sorted(bmp_name)

        for per in bmp_name:
            path_list.append(os.path.join(per_channel, '{}.bmp'.format(per)))

    return path_list

def save_path(base_path, path, label):
    save_path = os.path.join(base_path, str(label))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, 'path_dict.json'), 'w') as f:
        json.dump(path, f, indent=4)

if __name__ == '__main__':
    img_base_path = '/home/zhaomengjun/2021_binglang_paper/dataset/5_13/cut/cut_50'
    path_list = find_path(img_base_path)
    print(len(path_list))
    model_path = '/home/zhaomengjun/2021_binglang_paper/paper_code/save_model/cut_mobilenet_v2_epoch_205-acc_0.8544.pt'
    transforms = transform()
    label = 7
    path, acc = get_acc(path_list, model_path, transforms, label)

    base_path = '/home/zhaomengjun/2021_binglang_paper/data_path/cut'
    save_path(base_path, path, label)