from model.myResNet import resnet18
from model.BetelNet import BetelNet
import torchsummary
import cv2
import numpy as np
import os
import math
import time
import readjson
import torch
from cvtorchvision import cvtransforms
import torch.nn as nn
import psutil
from sklearn.externals import joblib

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_pca(pca_path, k):
    with open(pca_path, 'r') as f:
        js = json.load(f)
    mean = js['mean']
    std = js['std']
    pca_vec = js['feature'][:k]

    return mean, std, pca_vec

def Fillhole(thresh_img):
    thresh_img_c = thresh_img.copy()
    h, w = thresh_img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    isbreak = False
    for i in range(0, h):
        for j in range(0, w):
            if (thresh_img[i][j] == 0):
                seedpoint = (i, j)
                isbreak = True
                break
        if isbreak:
            break
    cv2.floodFill(thresh_img_c, mask, seedpoint, 255)
    thresh_img_inv = cv2.bitwise_not(thresh_img_c)
    fill_img = thresh_img | thresh_img_inv

    return fill_img

def delete_small(fill_img):
    img = fill_img.copy()
    binary, contours, hierarch = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # contours, hierarch = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    index = 0
    area = 0
    for i in range(0, len(contours)):
        cur_area = cv2.contourArea(contours[i])
        if cur_area > area:
            index = i
            area = cur_area

    for i in range(0, len(contours)):
        if i == index:
            continue
        else:
            cv2.drawContours(img, [contours[i]], -1, 0, thickness=-1)

    coor_tuple = cv2.boundingRect(contours[index])
    x1, y1 = coor_tuple[0], coor_tuple[1]
    x2, y2 = x1 + coor_tuple[2], y1 + coor_tuple[3]
    rect_img = img[y1:y2, x1:x2]

    return img, rect_img

def cal_angle(rect_img):
    rect_img1 = rect_img.copy()
    img_copy = rect_img.copy()
    middle_coor = []
    for i in range(0, rect_img.shape[0]):
        x = i
        j_list = []
        for j in range(0, rect_img.shape[1]):
            if rect_img[i][j] != 0:
                j_list.append(j)
        if len(j_list) == 0:
            continue
        else:
            y = j_list[len(j_list) // 2]
            middle_coor.append((y, x))

    middle_coor = np.array(middle_coor)
    line = cv2.fitLine(middle_coor, cv2.DIST_L2, 0, 0.01, 0.01)   #(cosa, sina, x, y)  (a: -90->90) 与x轴正向夹角，顺时针为正

    thea = np.arctan(line[1][0] / line[0][0])
    if thea < 0:
        rotate_angle = np.pi / 2 - abs(thea)
    else:
        rotate_angle = -(np.pi / 2 - thea)

    return rotate_angle

def rotate_img_f(img, angle):
    (h, w) = img.shape[:2]
    (cx, cy) = (w / 2, h /2 )
    M = cv2.getRotationMatrix2D((cx, cy), angle / np.pi * 180, 1.0)
    rotate_img = cv2.warpAffine(img, M, (w, h))

    return rotate_img

def get_shape_feature(img):
    _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    index = 0
    area = 0
    for i in range(0, len(contours)):
        cnt = contours[i]
        cur_area = cv2.contourArea(cnt)
        if cur_area > area:
            index = i
    cnt = contours[index]
    area = cv2.contourArea(cnt)                  #面积
    rect_para = cv2.boundingRect(cnt)
    rect_w, rect_h = rect_para[2], rect_para[3]   #宽、高
    scale_hw = rect_h / rect_w                        #高宽比
    length = cv2.arcLength(cnt, True)                   #周长
    area_scale = area / (rect_w * rect_h)                #轮廓面积/最小外接矩形面积
    hull_points = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull_points)
    area_hull = area / hull_area                            # 凸性
    Rc = 4 * np.pi * area / (length * length)                #圆度

    shape_features = [area, rect_w, rect_h, scale_hw, length, area_scale, area_hull, Rc]

    return rect_para, shape_features

def min_max_gray(cut_img):
    min_gray, max_gray = 256, 0
    h, w = cut_img.shape[0], cut_img.shape[1]
    for i in range(0, h):
        for j in range(0, w):
            if cut_img[i][j] > max_gray:
                max_gray = cut_img[i][j]
            if cut_img[i][j] < min_gray:
                min_gray = cut_img[i][j]
    return min_gray, max_gray

def set_gray(cut_img, min_gray, max_gray, gray_level):
    cut_img1 = cut_img.copy()
    for i in range(0, cut_img.shape[0]):
        for j in range(0, cut_img.shape[1]):
            cut_img1[i][j] = int((cut_img1[i][j] - min_gray) / (max_gray - min_gray) * (gray_level-1))
    return cut_img1

def get_glcm(cut_img, dx, dy, gray_level):
    ret = np.zeros((gray_level, gray_level))
    for i in range(0, cut_img.shape[0] - dy):
        for j in range(0, cut_img.shape[1] - dx):
            rows = cut_img[i][j]
            cols = cut_img[i+dy][j+dx]
            ret[rows][cols] += 1.0
    for i in range(0, gray_level):
        for j in range(0, gray_level):
            ret[i][j] /= float(cut_img.shape[0] * cut_img.shape[1])
    return ret

def get_feature(ret):
    energy, contrast, Idm, entropy = 0.0, 0.0, 0.0, 0.0
    for i in range(0, ret.shape[0]):
        for j in range(0, ret.shape[1]):
            energy += ret[i][j] * ret[i][j]               #能量
            contrast += (i - j) * (i - j) * ret[i][j]       #对比度
            Idm += ret[i][j] / (1 + (i - j) * (i - j))        #反差分矩阵
            if ret[i][j] > 0:
                entropy += ret[i][j] * math.log(ret[i][j])        #熵
    return energy, contrast, Idm, energy

def get_context_feature(cut_img):
    gray_level = 8
    min_gray, max_gray = min_max_gray(cut_img)
    cut_img1 = set_gray(cut_img, min_gray, max_gray, gray_level)
    ret = get_glcm(cut_img1, 1, 0, gray_level)
    energy, contrast, Idm, entropy =  get_feature(ret)

    context_features = [energy, contrast, Idm, entropy]
    return context_features

def fusion_feature(shape_features, context_features, mean, std, pca_vec):
    features = []
    for element in shape_features:
        features.append(element)
    for element in context_features:
        features.append(element)
    features = np.array(features)
    features = (features - mean) / std
    pca_vec = np.array(pca_vec)
    result = np.dot(features, np.transpose(pca_vec))
    result = list(result)

    return result

def betel(or_img, mean, std, pca_vec, model):
    t1 = time.time()
    crop_img = or_img[12:1012, 140:1140]
    gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    res_img = cv2.resize(gray_img, (224, 224))
    ret, thresh_img = cv2.threshold(res_img, 180, 255, cv2.THRESH_BINARY_INV)
    fill_img = Fillhole(thresh_img)
    img, rect_img = delete_small(fill_img)
    rotate_angle = cal_angle(rect_img)
    rotate_img = rotate_img_f(img, rotate_angle)
    rect_para, shape_features = get_shape_feature(rotate_img)

    rotate_img1 = rotate_img_f(res_img, rotate_angle)

    point1 = (rect_para[0] + rect_para[2] // 5, rect_para[1] + rect_para[3] // 3)
    point2 = (rect_para[0] + 4 * rect_para[2] // 5, rect_para[1] + 2 * rect_para[3] // 3)
    cut_img = rotate_img1[point1[1]:point2[1], point1[0]:point2[0]]

    # 获得纹理参数
    context_features = get_context_feature(cut_img)
    features_list = fusion_feature(shape_features, context_features, mean, std, pca_vec)
    t2 = time.time()
    # print('preprocess need {} ms'.format((t2-t1)*1000))

    vec = torch.Tensor(features_list)
    vec = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(vec, 0), 0), 0)
    vec = vec.permute(0, 3, 1, 2)
    vec = vec.to(torch.device('cuda'))
    out = model(vec)
    t3 = time.time()
    # print('betelnet need {} ms'.format((t3 - t1) * 1000))
    return (t3 - t1) * 1000, (t2 - t1) * 1000, features_list

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
    t4 = time.time()
    crop_img = or_img[12:1012, 140:1140]
    res_img = cv2.resize(crop_img, (224, 224))
    img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
    transform = transforms()
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    img = img.to(torch.device('cuda'))
    out = model(img)
    t5 = time.time()
    # print('resnet need {} ms'.format((t5-t4)*1000))

    return (t5-t4)*1000

def get_betel():
    pca_path = '/home/zhaomengjun/2021_binglang_paper/feature_dict/cluster_pca_stand.json'
    mean, std, pca_vec = get_pca(pca_path, 8)

    model_path = '/home/zhaomengjun/2021_binglang_paper/paper_code/logs/BetelNet/checkpoint/BN_prelu/201.pth'
    model = BetelNet(8, BN=True, f_flag='prelu')
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()

    return mean, std, pca_vec, model

def get_res():
    model_path = '/home/zhaomengjun/2021_binglang_paper/paper_code/logs/resnest18/checkpoint/618.pth'
    model = resnet18()
    if model_path is not None:
        checkpoint = torch.load(model_path)
        model_dict = model.state_dict()
        checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
    model.cuda()
    model.eval()
    return model

def get_svm():
    svm_path = '/home/zhaomengjun/2021_binglang_paper/paper_code/logs/svm/svm.m'
    cls = joblib.load(svm_path)

    return cls

def svm(features_list, model):
    svm_t1 = time.time()
    features = np.array(features_list)
    features = features.reshape(1, -1)
    predict_label = model.predict(features)
    svm_t2 = time.time()
    # print('svm need {} ms'.format((svm_t2 - svm_t1) * 1000))

    return (svm_t2 - svm_t1) * 1000

if __name__ == '__main__':
    # model = BetelNet(8).cuda()
    # torchsummary.summary(model, (8,1,1))
    # print('parameters_count:', count_parameters(model))

    img_path = '/home/zhaomengjun/2021_binglang_paper/dataset_end/cluster_con/4/4_4_1.bmp'
    or_img = cv2.imread(img_path)

    mean, std, pca_vec, betel_model = get_betel()
    torchsummary.summary(betel_model, (8,1,1))
    print('betel parameters_count:', count_parameters(betel_model))
    _, _, features_list = betel(or_img,  mean, std, pca_vec, betel_model)

    res_model = get_res()
    torchsummary.summary(res_model, (3,224,224))
    print('resnet parameters_count:', count_parameters(res_model))
    resnet(or_img, res_model)

    svm_model = get_svm()
    svm(features_list, svm_model)

    bet_all_time, res_all_time, svm_all_time = 0, 0, 0
    for i in range(100):
        bet_time, pre_time,  features_list = betel(or_img, mean, std, pca_vec, betel_model)
        res_time = resnet(or_img, res_model)
        svm_time = svm(features_list, svm_model) + pre_time
        bet_all_time += bet_time
        res_all_time += res_time
        svm_all_time += svm_time
    print('betlnet need {} ms'.format(bet_all_time / 100))
    print('resnet need {} ms'.format(res_all_time / 100))
    print('svm need {} ms'.format(svm_all_time / 100))

    print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))