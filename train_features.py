import json
import os
import torch
from data.dataset import MyDataset
from model import BetelNet
from cfg.cfg import parameters as cfg
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
from tool import loss

def run_epoch(model, device,  trainloader, testloader, epochs, loss_func, optimizer, scheduler, writer = None, flag=None):
    max_acc = [0, 0]
    loss_dict = {'train': [], 'test': []}
    acc_dict = {'train': [], 'test': []}
    for epoch in range(epochs):
        train_loss = 0
        train_correct = 0
        train_num = 0
        test_loss = 0
        test_correct = 0
        test_num = 0
        for data, label in trainloader:
            model.train()
            optimizer.zero_grad()

            data = data.to(device)
            label = label.to(device)

            out = model(data)

            loss = loss_func(out, label)
            pred = torch.max(out, 1)[1]

            train_loss += loss.item()
            train_correct += ((pred == label).sum()).item()

            train_num += len(data)

            loss.backward()
            optimizer.step()

        train_loss = train_loss / train_num
        train_acc = train_correct / train_num
        lr = optimizer.param_groups[0]['lr']

        print('[{} {} {}] : {} {} {}'.format(flag, 'train', epoch, train_loss, train_acc, lr))

        # if writer is not None:
        #     writer.add_scalar('epoch/{}_train_loss'.format(flag), train_loss, epoch)
        #     writer.add_scalar('epoch/{}_train_acc'.format(flag), train_acc, epoch)
        #     writer.add_scalar('epoch/lr', lr, epoch)
        #     writer.flush()

        for data, label in testloader:
            model.eval()
            data = data.to(device)
            label = label.to(device)
            out = model(data)
            loss = loss_func(out, label)
            pred = torch.max(out, 1)[1]
            test_loss += loss.item()
            test_correct += ((pred == label).sum()).item()
            test_num += len(data)
        test_loss = test_loss / test_num
        test_acc = test_correct / test_num
        lr = optimizer.param_groups[-1]['lr']
        if max_acc[0] < test_acc:
            max_acc[0] = test_acc
            max_acc[1] = epoch
        print('[{} {} {}] : {} {} {}'.format(flag, 'test', epoch, test_loss, test_acc, lr))

        # if writer is not None:
        #     writer.add_scalar('epoch/{}_test_loss'.format(flag), test_loss, epoch)
        #     writer.add_scalar('epoch/{}_test_acc'.format(flag), test_acc, epoch)
        #     writer.flush()

        # save_base_pth = os.path.join('/home/zhaomengjun/2021_binglang_paper/paper_code/logs/BetelNet/checkpoint', flag)
        # if not os.path.exists(save_base_pth):
        #     os.makedirs(save_base_pth)
        # save_path = os.path.join(save_base_pth, '{}.pth'.format(epoch))
        # torch.save(model.state_dict(), save_path)

        scheduler.step()
        # scheduler.step(test_acc)
        acc_dict['train'].append(train_acc)
        acc_dict['test'].append(test_acc)
        loss_dict['train'].append(train_loss)
        loss_dict['test'].append(test_loss)
    print(max_acc)
    save_path = '/home/zhaomengjun/2021_binglang_paper/paper_code/logs/BetelNet'
    with open(os.path.join(save_path, '{}_acc.json'.format(flag)), 'w') as f:
        json.dump(acc_dict, f, indent=4)
    with open(os.path.join(save_path, '{}_loss.json'.format(flag)), 'w') as f:
        json.dump(loss_dict, f, indent=4)

    return acc_dict['test'], loss_dict['test']

def train(trainloader, testloader, acc_dict, loss_dict, BN=False, f_flag='prelu'):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    model = BetelNet(cfg['in_channels'], BN, f_flag)
    model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(),lr=cfg['lr'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    loss_func = loss.LabelSmoothingCrossEntropy()

    epochs = cfg['epochs']
    if BN:
        flag = '{}_{}'.format('BN', f_flag)
    else:
        flag = f_flag

    logs_path = '/home/zhaomengjun/2021_binglang_paper/paper_code/logs/BetelNet/shape_texture'
    run_path = os.path.join(logs_path,f_flag)
    if not os.path.exists(run_path):
        os.makedirs(run_path)

    writer = SummaryWriter(run_path)

    test_acc, test_loss =  run_epoch(model, device, trainloader, testloader, epochs, loss_func, optimizer, scheduler, writer, flag)
    print('{} end'.format(flag))

    acc_dict[flag] = test_acc
    loss_dict[flag] = test_loss

    return acc_dict, loss_dict

def save_json(save_path, acc_dict, loss_dict, flag=None):
    with open(os.path.join(save_path, flag+'_acc'+'.json'), 'w') as f:
        json.dump(acc_dict, f, indent=4)
    with open(os.path.join(save_path, flag+'_loss'+'.json'), 'w') as f:
        json.dump(loss_dict, f, indent=4)

if __name__ == '__main__':
    train_path = '/home/zhaomengjun/2021_binglang_paper/feature_dict/cluster_train_data.json'
    test_path = '/home/zhaomengjun/2021_binglang_paper/feature_dict/cluster_test_data.json'
    save_path = '/home/zhaomengjun/2021_binglang_paper/paper_code/logs/BetelNet/compare'

    trainset = MyDataset(train_path)
    testset = MyDataset(test_path)

    trainloader = DataLoader(dataset=trainset,
                             batch_size=cfg['batch_size'],
                             shuffle=True,
                             pin_memory=True
                             )

    testloader = DataLoader(dataset=testset,
                             batch_size=cfg['batch_size'],
                             shuffle=True,
                             pin_memory=True
                             )


    acc_dict = {}
    loss_dict = {}

    train(trainloader, testloader,  acc_dict, loss_dict, BN=True, f_flag='prelu')

    # flag = 'f'
    # if flag == 'f':
    #     f_flag = ['swish', 'prelu', 'relu']
    #     for f in f_flag:
    #         acc_dict, loss_dict = train(trainloader, testloader, acc_dict, loss_dict, BN=True, f_flag=f)
    #     save_json(save_path, acc_dict, loss_dict, flag='bn_'+flag)
    # elif flag == 'bn':
    #     f_flag = 'prelu'
    #     bn_flag = [True, False]
    #     for bn in bn_flag:
    #         acc_dict, loss_dict = train(trainloader, testloader, acc_dict, loss_dict, BN=bn, f_flag=f_flag)
    #     save_json(save_path, acc_dict, loss_dict, flag=flag)

