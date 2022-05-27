import math
import os
import json
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
from pylab import mpl

def open_json(read_path):
    with open(read_path, 'r') as f:
        js = json.load(f)
    train_list = js['BN_swish']
    test_list = js['BN_prelu']
    test_list1 = js['BN_relu']

    return train_list, test_list, test_list1

def draw_plot(train_list, test_list,test_list1,flag, save_path):
    fig = plt.figure(figsize=(5.5, 4))
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)
    ax.axis["bottom"].set_axisline_style("->", size=1.5)
    ax.axis["left"].set_axisline_style("->", size=1.5)
    ax.axis["top"].set_visible(False)
    ax.axis["right"].set_visible(False)

    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    x = [i for i in range(len(train_list))]
    plt.plot(x, train_list, label='swish')
    plt.plot(x, test_list, label='prelu')
    plt.plot(x, test_list1, label='relu')
    plt.xlabel('Epoch(迭代次数)', fontsize=30)
    plt.ylabel(flag, fontsize=30)
    plt.legend()
    plt.savefig(os.path.join(save_path, flag+'.jpg'))
    plt.show()

if __name__ == '__main__':
    base_path = r'E://vis_compare'
    train_acc, test_acc, test_acc1 = open_json(os.path.join(base_path, '1bn_f_acc.json'))
    train_loss, test_loss, test_loss1 = open_json(os.path.join(base_path, '1bn_f_loss.json'))

    draw_plot(train_acc, test_acc,test_acc1, 'Accuracy(准确率)', base_path)
    draw_plot(train_loss, test_loss, test_loss1, 'Loss(损失)', base_path)