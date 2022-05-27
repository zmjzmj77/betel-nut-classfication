import torch.cuda

from old_model.myResNet import ResNet18
from model.myResNet import resnet18
import torch.nn as nn
from cfg.cfg import parameters as cfg
from data.res_dataset import res_Dataset
from torch.utils.data import DataLoader
from cfg.cfg import parameters as cfg
from tool import loss
import time
import os
from torch.utils.tensorboard import SummaryWriter
from tool.tom import Tom
from tool.loss import mix_loss
import numpy as np
from warmup_scheduler import GradualWarmupScheduler
import json

from old_model.myVIT.vit import ViT
from model.ConvNet import ConvNet

from tool.regular import Regularization

def create_model(resnet, num_class, pretrain_path = None):
    model = resnet()
    if pretrain_path is not None:
        checkpoint = torch.load(pretrain_path)
        model_dict = model.state_dict()
        checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
    in_features = model.new_fc.in_features
    model.new_fc = nn.Linear(in_features, num_class, bias=True)
    return model

def run_epochs(model, trainloader, testloader, epochs, loss_func, optimizer, scheduler,
               scheduler_warmup, writer = None, adjust = False, reg_loss = None):
    loss_dict = {'train': [], 'test': []}
    acc_dict = {'train': [], 'test': []}
    max_acc = [0, 0]
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

            data = data.cuda()
            label = label.cuda()
            if cfg['data_aug']:
                batch = data.size()[0]
                index = torch.randperm(batch).cuda()
                lam = np.random.beta(1.0, 1.0)
                mixed_data = lam * data + (1 - lam) * data[index, :]
                out = model(mixed_data)
                label1, label2 = label, label[index]
                loss = mix_loss(loss_func, out, label1, label2, lam)
            else:
                out = model(data)
                loss = loss_func(out, label)
                if reg_loss is not None:
                    loss += reg_loss(model)
            pred = torch.max(out, 1)[1]

            train_loss += loss.item()
            train_correct += ((pred == label).sum()).item()

            train_num += len(data)

            loss.backward()
            optimizer.step()

        train_loss = train_loss / train_num
        train_acc = train_correct / train_num
        lr = optimizer.param_groups[0]['lr']

        print('[{} {}] : {} {} {}'.format('train', epoch, train_loss, train_acc, lr))

        if writer is not None:
            writer.add_scalar('epoch/train_loss', train_loss, epoch)
            writer.add_scalar('epoch/train_acc', train_acc, epoch)
            writer.add_scalar('epoch/lr', lr, epoch)
            writer.flush()

        for data, label in testloader:
            model.eval()

            data = data.cuda()
            label = label.cuda()

            out = model(data)

            loss = loss_func(out, label)
            pred = torch.max(out, 1)[1]

            test_loss += loss.item()
            test_correct += ((pred == label).sum()).item()

            test_num += len(data)

        test_loss = test_loss / test_num
        test_acc = test_correct / test_num
        lr = optimizer.param_groups[0]['lr']
        if max_acc[0] < test_acc:
            max_acc[0] = test_acc
            max_acc[1] = epoch

        save_path = os.path.join('/home/zhaomengjun/2021_binglang_paper/paper_code/logs/resnet18/checkpoint', str(epoch)+'.pth')
        torch.save(model.module.state_dict(), save_path)
        print('[{} {}] : {} {} {}'.format('test', epoch, test_loss, test_acc, lr))

        if writer is not None:
            writer.add_scalar('epoch/test_loss', test_loss, epoch)
            writer.add_scalar('epoch/test_acc', test_acc, epoch)
            writer.flush()

        if adjust:
            # scheduler.step(test_acc)
            scheduler_warmup.step(test_acc)
        else:
            # scheduler.step()
            scheduler_warmup.step()

        acc_dict['test'].append(test_acc)
        acc_dict['train'].append(train_acc)
        loss_dict['train'].append(train_loss)
        loss_dict['test'].append(test_loss)
    print(max_acc)
    save_path = '/home/zhaomengjun/2021_binglang_paper/paper_code/logs/resnet18'
    with open(os.path.join(save_path, 'acc.json'), 'w') as f:
        json.dump(acc_dict, f, indent=4)
    with open(os.path.join(save_path, 'loss.json'), 'w') as f:
        json.dump(loss_dict, f, indent=4)

if __name__ == '__main__':
    train_path = '/home/zhaomengjun/2021_binglang_paper/dataset_end/224_con/train.json'
    test_path = '/home/zhaomengjun/2021_binglang_paper/dataset_end/224_con/test.json'

    # cfg['pretrain_path'] = None
    # model = create_model(resnet18, cfg['num_class'], cfg['pretrain_path'])
    # for k, v in model.named_parameters():
    #     if k.split('.')[0]  in  ['new_layer1', 'new_layer2','new_layer3', 'new_layer4', 'new_fc', 'layer5_1', 'layer5_2'] or k.split('.')[-2] in ['conv7']:
    #         v.requires_grad = True
    #     else:
    #         v.requires_grad = False
    #     print(k, v.requires_grad)

    #resnet18
    model = create_model(ResNet18, cfg['num_class'], cfg['pretrain_path'])
    for k, v in model.named_parameters():
        if k.split('.')[0]  in  ['conv1', 'bn1', 'layer1', 'layer2']:
            v.requires_grad = False
        else:
            v.requires_grad = True
        print(k, v.requires_grad)

    print('model load sucess')

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.cuda()

    data_aug = cfg['data_aug']
    trainset = res_Dataset(train_path, dataaug=data_aug)
    testset = res_Dataset(test_path)

    trainloader = DataLoader(dataset=trainset,
                             batch_size=cfg['batch_size'],
                             shuffle=True,
                             num_workers=8,
                             pin_memory=True)

    testloader = DataLoader(dataset=testset,
                            batch_size=cfg['batch_size'],
                            shuffle=True,
                            num_workers=8,
                            pin_memory=True)
    print("dataset success")

    loss_func = loss.LabelSmoothingCrossEntropy()

    reg_loss = None

    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=cfg['lr']
                                 )

    # pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = Tom(pg, lr = cfg['lr'], weight_decay=5e-5)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.975)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5,
    #                                                        patience=5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.75)

    scheduler_warmup = GradualWarmupScheduler(optimizer,
                                              multiplier=1.0,
                                              total_epoch=20,
                                              after_scheduler=scheduler)

    cur_time = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
    logs_path = '/home/zhaomengjun/2021_binglang_paper/paper_code/logs/resnet18'
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    run_path = os.path.join(logs_path, 'cur_time')
    if not os.path.exists(run_path):
        os.makedirs(run_path)

    writer = SummaryWriter(run_path)

    epochs = cfg['epochs']

    print('start train')
    run_epochs(model, trainloader, testloader, epochs, loss_func, optimizer, scheduler, scheduler_warmup,
               writer, adjust = False, reg_loss = reg_loss)