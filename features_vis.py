import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import style
import mpl_toolkits.axisartist as axisartist
from pylab import mpl
import math

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
    # cv2.imshow('floodfill', thresh_img_c)
    thresh_img_inv = cv2.bitwise_not(thresh_img_c)

    fill_img = thresh_img | thresh_img_inv
    # cv2.imshow('fill', fill_img)

    return fill_img

def delete_small(fill_img):
    img = fill_img.copy()
    # binary, contours, hierarch = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours, hierarch = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

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

def cal_angle(rect_img, path):
    cv2.imwrite(os.path.join(path, 'rect.jpg'), rect_img)
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
    # print(rect_img.shape[0], rect_img.shape[1], len(middle_coor))
    for per_point in middle_coor:
        cv2.circle(rect_img1, per_point, 1, 100, 1)
    cv2.imshow('middle points', rect_img1)
    cv2.imwrite(os.path.join(path, 'middle.bmp'), rect_img1)

    middle_coor = np.array(middle_coor)
    line = cv2.fitLine(middle_coor, cv2.DIST_L2, 0, 0.01, 0.01)   #(cosa, sina, x, y)  (a: -90->90) 与x轴正向夹角，顺时针为正
    # print(line)

    k = line[1][0] / line[0][0]
    b = line[3][0] - k * line[2][0]
    h, w = rect_img.shape[0], rect_img.shape[1]
    point1 = (int((0 - b) / k), 0)
    point2 = (int((h - b) // k), h)
    cv2.line(img_copy, point1, point2, 150, 1, 4)
    cv2.line(img_copy, (w//2, 0), ( w//2, h), 150, 1, 4)
    cv2.imshow('fit line', img_copy)
    cv2.imwrite(os.path.join(path, 'fit_line.bmp'), img_copy)

    thea = np.arctan(line[1][0] / line[0][0])
    # print(thea)
    if thea < 0:
        rotate_angle = np.pi / 2 - abs(thea)
    else:
        rotate_angle = -(np.pi / 2 - thea)

    return rotate_angle

def rotate_img_f(img, angle, path, flag):
    (h, w) = img.shape[:2]
    (cx, cy) = (w / 2, h /2 )
    M = cv2.getRotationMatrix2D((cx, cy), angle / np.pi * 180, 1.0)
    # print(M)

    rotate_img = cv2.warpAffine(img, M, (w, h))
    cv2.imshow('rotate'+flag, rotate_img)
    cv2.imwrite(os.path.join(path, 'rotate_{}.bmp'.format(flag)), rotate_img)

    return rotate_img

def plot(res_img):
    # 创建画布
    fig = plt.figure(figsize=(5,4))
    # 使用axisartist.Subplot方法创建一个绘图区对象ax
    ax = axisartist.Subplot(fig, 111)
    # 将绘图区对象添加到画布中
    fig.add_axes(ax)
    # 通过set_axisline_style方法设置绘图区的底部及左侧坐标轴样式
    # "-|>"代表实心箭头："->"代表空心箭头
    ax.axis["bottom"].set_axisline_style("->", size=1.5)
    ax.axis["left"].set_axisline_style("->", size=1.5)
    # 通过set_visible方法设置绘图区的顶部及右侧坐标轴隐藏
    ax.axis["top"].set_visible(False)
    ax.axis["right"].set_visible(False)

    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    hist, bins = np.histogram(res_img.ravel(), 256, [0, 250])
    plt.plot(hist, c='b')
    # plt.grid(b=False)
    plt.xlabel('灰度值', fontsize=15)
    plt.ylabel('像素个数',fontsize=15)
    # plt.axis('off')
    plt.axvline(x=180, c='r')
    plt.text(185,50,'灰度值=180',c='r',fontsize=11)
    plt.savefig(os.path.join(path, 'hist.jpg'))
    plt.show()

def get_shape_feature(img):
    # _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)   #linux
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)      #windows

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

    return rect_para, shape_features, img
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
    # if max_gray - min_gray + 1 > gray_level:
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

if __name__ == '__main__':
    read_path = r'E:\bmp'
    path = r'E:\preprocess'
    for img_path in glob.glob('{}/*'.format(read_path)):
        if os.path.basename(img_path) != '124.bmp':
            continue
        print(os.path.basename(img_path))
        or_img = cv2.imread(img_path)
        crop_img = or_img[12:1012, 140:1140]
        res_img = cv2.resize(crop_img, (224, 224))
        cv2.imshow('ori', res_img)
        cv2.imwrite(os.path.join(path, 'ori.bmp'), res_img)

        gray_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray', gray_img)
        cv2.imwrite(os.path.join(path, 'gray.bmp'), gray_img)

        # plot(res_img)

        # 二值化
        ret, thresh_img = cv2.threshold(gray_img, 180, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow('hsit', thresh_img)
        cv2.imwrite(os.path.join(path, 'thresh.bmp'), thresh_img)

        # 孔洞填充
        fill_img = Fillhole(thresh_img)
        cv2.imshow('fill', fill_img)
        cv2.imwrite(os.path.join(path, 'fill.bmp'), fill_img)
        # 保留最大的区域
        img, rect_img = delete_small(fill_img)
        cv2.imshow('cut_small', img)
        cv2.imwrite(os.path.join(path, 'cut_small.bmp'), img)
        # 计算旋转角度
        rotate_angle = cal_angle(rect_img, path)
        # 旋转二值图，并获得形状参数
        rotate_img = rotate_img_f(img, rotate_angle, path, 'thresh')

        rect_para, shape_features, rect_nut_img = get_shape_feature(rotate_img)

        point1 = (rect_para[0], rect_para[1])
        point2 = (rect_para[0] + rect_para[2], rect_para[1] + rect_para[3])
        cv2.rectangle(rect_nut_img, point1, point2, 100, 1)
        cv2.imwrite(os.path.join(path, 'rect_nut.bmp'), rect_nut_img)
        # 旋转灰度图
        rotate_img1 = rotate_img_f(gray_img, rotate_angle,path, 'gray')

        point1 = (rect_para[0] + rect_para[2] // 5, rect_para[1] + rect_para[3] // 3)
        point2 = (rect_para[0] + 4 * rect_para[2] // 5, rect_para[1] + 2 * rect_para[3] // 3)
        cv2.rectangle(rotate_img1, point1, point2, (200, 200 , 200), 1)
        cv2.imwrite(os.path.join(path, 'rect_gray.bmp'), rotate_img1)
        cut_img = rotate_img1[point1[1]:point2[1], point1[0]:point2[0]]
        cv2.imshow('cut_gray', cut_img)
        cv2.imwrite(os.path.join(path, 'cut_gray.bmp'), cut_img)

        # 获得纹理参数z
        context_features = get_context_feature(cut_img)

        cv2.waitKey()
