import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
import torch as ch
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms
from advertorch.attacks import L2PGDAttack,LinfPGDAttack,FGSM

import os
import copy
import datetime
import shutil

from models import *
from utils_AT import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#net = ResNet18()
net = WRN32_10_GNN(block_pos=[1,2,3])

with open('configs_simple.yml') as f:
    config = EasyDict(yaml.load(f))

eps_record = config.ADV.clip_eps
step_record = config.ADV.fgsm_step
file_name = config.Operation.Prefix
record_words = config.Operation.record_words+"\n"
check_path = os.path.join('./checkpoint',file_name)
record_path = os.path.join(check_path, file_name+'_record.txt')
resume = False
learning_rate = 0.1

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

def save_checkpoint(state, is_best, filepath):
    filename = os.path.join(filepath, 'checkpoint.pth.tar')
    # Save model
    torch.save(state, filename)
    # Save best model
    if is_best:
        shutil.copyfile(filename, os.path.join(filepath, 'model_best.pth.tar'))

def train(epoch):
    print('\n[ Train epoch: %d ]' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        benign_outputs = net(inputs)
        loss = criterion(benign_outputs, targets)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        _, predicted = benign_outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('\nTotal benign train accuarcy:', 100. * correct / total)
    print('Total benign train loss:', train_loss)

    return 100. * correct / total, train_loss

def test(epoch, best_prec, save_path='./checkpoint'):
    print('\n[ Test epoch: %d ]' % epoch)
    net.eval()
    benign_loss = 0
    benign_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            benign_loss += loss.item()

            _, predicted = outputs.max(1)
            benign_correct += predicted.eq(targets).sum().item()

    test_acc = 100. * benign_correct / total
    print('\nTotal benign test accuarcy:', test_acc)
    is_best = test_acc > best_prec
    best_prec1 = max(test_acc,best_prec)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_checkpoint({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, os.path.join(save_path))
    print('Model Saved!')
    
    return 100. * benign_correct / total, benign_loss,  best_prec1

def test_pgd(net,adversary,test_loader):
    net.eval()
    adv_correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)
        adv = adversary.perturb(inputs, targets)
        adv_outputs = net(adv)
        _, predicted = adv_outputs.max(1)
        adv_correct += predicted.eq(targets).sum().item()

    adv_acc = 100. * adv_correct / total
    return adv_acc

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

net = net.to(device)
net = torch.nn.DataParallel(net)  # parallel GPU
cudnn.benchmark = True

if config.Operation.Resume == True:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(check_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(check_path,'checkpoint.pth.tar'))
    net.load_state_dict(checkpoint['state_dict'])
    start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
else:
    start_epoch = 0
    best_prec1 = 0
    if not os.path.isdir(check_path):
        os.mkdir(check_path)
    with open(record_path,"w+") as f:
        f.write(record_words)
        date_Ex = ('Experiment Date: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))
        f.write(date_Ex+'\n')
    f.close()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002)


for epoch in range(start_epoch+1, 201):
    adjust_learning_rate(optimizer, epoch)
    acc_train, loss_train = train(epoch)
    acc_test,  loss_test, best_prec1 = test(epoch, best_prec1, save_path=check_path)
    with open(record_path,"a+") as f:
        f.write("epoch:"+str(epoch)+" "+f"acc_train:{acc_train:.2f}, acc_test:{acc_test:.2f}"+"\n")
    f.close()

if config.Operation.Validate == True:
    #TODO: PGD Attack test
    print("==> Loading best model:"+file_name+"\n")
    assert os.path.isdir(check_path), 'Error: no checkpoint directory found!'
    checkpoint_best = torch.load(os.path.join(check_path, 'model_best.pth.tar'))

    with open(record_path, "a+") as f:
        f.write("Best_model Performance: \n")
    f.close()
    net.load_state_dict(checkpoint_best['state_dict'])
    adversary_1 = LinfPGDAttack(net,loss_fn=nn.CrossEntropyLoss(),eps=eps_record/255.0,nb_iter=config.ADV.pgd_attack_1,eps_iter=step_record/255.0,
                              rand_init=True,clip_min=0.0,clip_max=1.0)
    adversary_2 = LinfPGDAttack(net,loss_fn=nn.CrossEntropyLoss(),eps=eps_record/255.0,nb_iter=config.ADV.pgd_attack_2,eps_iter=step_record/255.0,
                              rand_init=True,clip_min=0.0,clip_max=1.0)
    for adv_attack in [adversary_1, adversary_2]:
        test_acc_pgd = test_pgd(net,adv_attack,test_loader)
        print(f"PGD_attack:[nb_iter:{adv_attack.nb_iter},eps:{eps_record},step_size:{step_record}]->pgd_acc: {test_acc_pgd: .2f}" + "\n")
        with open(record_path, "a+") as f:
            f.write(f"PGD_attack:[nb_iter:{adv_attack.nb_iter},eps:{eps_record},step_size:{step_record}]->pgd_acc: {test_acc_pgd: .2f}" + "\n")
        f.close()
