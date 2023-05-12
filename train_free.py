import os
import time
import sys
import math
import numpy as np
import yaml
import datetime

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from easydict import EasyDict

from models import *
from utils_AT import *
from validation import validate, validate_pgd, validate_fgsm, record_path_words,validate_autoattack

# Load Parameter
with open('configs.yml') as f:
    config = EasyDict(yaml.load(f))

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])


def main():
    net = WRN32_10_GNN(block_pos=[1,2,3])
    # net = WRN32_10()

    output_prefix = config.Operation.Prefix
    output_name = '{:s}_step{:d}_eps{:d}_repeat{:d}'.format(output_prefix,config.ADV.fgsm_step,
                                                            config.ADV.clip_eps,config.ADV.n_repeats)
    eps_record = config.ADV.clip_eps
    step_record = config.ADV.fgsm_step
    # Scale and initialize the parameters
    best_prec1 = 0
    config.TRAIN.epochs = int(math.ceil(config.TRAIN.epochs / config.ADV.n_repeats))
    config.ADV.fgsm_step /= config.DATA.max_color_value
    config.ADV.clip_eps /= config.DATA.max_color_value
    date_Ex = ('Experiment Date: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))


    # Create output folder
    if not os.path.isdir(os.path.join('./checkpoint', output_name)):
        os.makedirs(os.path.join('./checkpoint', output_name))
    record_path = os.path.join('./checkpoint', output_name, output_prefix+'_record.txt')

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)
    
    # Model:
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

    # Criterion:
    criterion = nn.CrossEntropyLoss().cuda()
    
    # Optimizer:
    optimizer = torch.optim.SGD(net.parameters(), config.TRAIN.lr,
                                momentum=config.TRAIN.momentum,
                                weight_decay=config.TRAIN.weight_decay)
    
    if config.Operation.Train == True:
    # Resume if a valid checkpoint path is provided
        if config.Operation.Resume == True:
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('./checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(os.path.join('./checkpoint',output_name,'checkpoint.pth.tar'))
            net.load_state_dict(checkpoint['state_dict'])
            config.TRAIN.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                      .format(output_prefix, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(output_prefix))
        if config.Operation.Resume == False:
            with open(record_path, "w+") as f:
                f.write(date_Ex + ' ' + output_prefix +' step_size:'+ str(step_record)+
                        ' eps:'+ str(eps_record) +' n_iters:'+ str(config.ADV.n_repeats) +'\n')
            f.close()
        for epoch in range(config.TRAIN.start_epoch, config.TRAIN.epochs):
            adjust_learning_rate(config.TRAIN.lr, optimizer, epoch, config.ADV.n_repeats)

            # train for one epoch
            train_acc, train_loss =train(train_loader, net, criterion, optimizer, epoch)

            # evaluate on validation set
            test_acc = validate(test_loader, net, criterion, config)

            # remember best prec@1 and save checkpoint
            is_best = test_acc > best_prec1
            best_prec1 = max(test_acc, best_prec1)
            if not os.path.isdir(os.path.join('./checkpoint', output_name)):
                os.mkdir(os.path.join('./checkpoint', output_name))
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, os.path.join('./checkpoint',output_name))
            with open(record_path, "a+") as f:
                f.write("epoch:" + str(epoch) + " " + f"acc_train:{train_acc.item():.2f}, acc_test:{test_acc.item():.2f}" + "\n")
            f.close()
            record_path_words(record_path, f"Best_acc_test:{best_prec1:.2f} " + "\n")

    if config.Operation.Validate == True:
        print("==> Loading best model: "+output_prefix+"\n")
        assert os.path.isdir(os.path.join('./checkpoint', output_name)), 'Error: no checkpoint directory found!'
        checkpoint_best = torch.load(os.path.join('./checkpoint', output_name, 'model_best.pth.tar'))
        net.load_state_dict(checkpoint_best['state_dict'])
        record_path_words(record_path,"Best_model Performance: \n")
        test_acc_fgsm = validate_fgsm(test_loader, net, criterion, eps_record, config)
        record_path_words(record_path, f"FGSM_attack:[eps:{eps_record}->fgsm_acc: {test_acc_fgsm.item(): .2f}" + "\n")
        for pgd_param in config.ADV.pgd_attack:
            test_acc_pgd = validate_pgd(test_loader, net, criterion, pgd_param[0], pgd_param[1], config)
            record_path_words(record_path, f"PGD_attack:[nb_iter:{pgd_param[0]},eps:{eps_record},step_size:{step_record}]->pgd_acc: {test_acc_pgd.item(): .2f}" + "\n")
        net.load_state_dict(checkpoint_best['state_dict'])
        test_acc_AA = validate_autoattack(test_loader, net, eps_record, config)
        record_path_words(record_path, f"AA_attack:[eps:{eps_record}->AA_acc: {test_acc_AA: .2f}" + "\n")


# Free Adversarial Training Module        
global global_noise_data
#TODO: 修改
global_noise_data = torch.zeros([config.DATA.batch_size, 3, config.DATA.crop_size, config.DATA.crop_size]).cuda()
def train(train_loader, model, criterion, optimizer, epoch):
    global global_noise_data
    mean = torch.Tensor(np.array(config.TRAIN.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3,config.DATA.crop_size, config.DATA.crop_size).cuda()
    std = torch.Tensor(np.array(config.TRAIN.std)[:, np.newaxis, np.newaxis])
    std = std.expand(3, config.DATA.crop_size, config.DATA.crop_size).cuda()
    # Initialize the meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()
    for i, (input, target) in enumerate(train_loader):
        end = time.time()
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        data_time.update(time.time() - end)
        for j in range(config.ADV.n_repeats):
            # Ascend on the global noise
            noise_batch = Variable(global_noise_data[0:input.size(0)], requires_grad=True).cuda()
            in1 = input + noise_batch
            in1.clamp_(0, 1.0)  #in-place类型: 直接修改了这个tensor，还是返回一个新的tensor，而旧的tensor并不修改。
            in1.sub_(mean).div_(std)
            output = model(in1)
            loss = criterion(output, target)
            
            prec1,prec5 = accuracy(output, target, topk=(1,5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            
            # Update the noise for the next iteration
            pert = fgsm(noise_batch.grad, config.ADV.fgsm_step)
            global_noise_data[0:input.size(0)] += pert.data
            global_noise_data.clamp_(-config.ADV.clip_eps, config.ADV.clip_eps)

            optimizer.step()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % config.TRAIN.print_freq == 0:
                print(f'Train Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                      f'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                      f'Data {data_time.val:.2f} ({data_time.avg:.2f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Prec@1 {top1.val.item():.3f} ({top1.avg.item():.3f})\t'
                      f'Prec@5 {top5.val.item():.3f} ({top5.avg.item():.3f})\t')
                sys.stdout.flush()
    return top1.avg, losses.avg

if __name__ == '__main__':
    main()
