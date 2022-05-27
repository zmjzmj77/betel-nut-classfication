import torch
import cv2
from model.BetelNet import BetelNet
import os

def tracec_model(model_path, pt_path):
    print('start')
    img = torch.randn(1,8,1,1)
    img = img.cuda()
    model = BetelNet(8, BN=True, f_flag='prelu')
    model.load_state_dict(torch.load(model_path))
    # model = torch.load(model_path)
    model = torch.nn.DataParallel(model)
    model.cuda()
    model.eval()
    trace_module = torch.jit.trace(model.module, img)
    trace_module.save(pt_path)
    print('end')


if __name__ == '__main__':
    model_path = '/home/zhaomengjun/2021_binglang_paper/paper_code/logs/BetelNet/checkpoint/BN_prelu/201.pth'
    pt_path = '/home/zhaomengjun/2021_binglang_paper/paper_code/logs/BetelNet/checkpoint/betelnet.pt'
    tracec_model(model_path, pt_path)