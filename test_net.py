import os.path
import shutil

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms
from autoattack import AutoAttack

from models import *
from advertorch.attacks import L2PGDAttack,LinfPGDAttack,FGSM
import yaml
from easydict import EasyDict

device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = WRN32_10()

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

def test_adv(net,adversary,test_loader):
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

def test_adv_auto(net, adversary, test_loader):
    test_images = []
    test_labels = []

    for images, labels in test_loader:
        test_images.append(images)
        test_labels.append(labels)

    test_images = torch.cat(test_images, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    net.eval()
    inputs, targets = test_images.to(device), test_labels.to(device)
    adv = adversary.run_standard_evaluation(inputs, targets, bs=250)

    # Compute adversarial outputs in batches
    adv_outputs = []
    total = len(adv)
    batch_size = 100
    for i in range(0, total, batch_size):
        adv_batch = adv[i:i+batch_size]
        adv_outputs_batch = net(adv_batch)
        adv_outputs.append(adv_outputs_batch.cpu().detach().numpy())
    adv_outputs = np.concatenate(adv_outputs, axis=0)

    # Compute adversarial accuracy
    _, predicted = torch.max(torch.from_numpy(adv_outputs), 1)
    predicted = predicted.to(device)
    adv_correct = predicted.eq(targets).sum().item()
    return 100. * adv_correct / targets.size(0)

def record_path_words(record_path, record_words):
    print(record_words)
    with open(record_path, "a+") as f:
        f.write(record_words)
    f.close()
    return 

with open('configs_test.yml') as f:
    config = EasyDict(yaml.load(f))

eps_record = config.ADV.clip_eps
step_record = config.ADV.fgsm_step
file_name = config.Operation.Prefix
check_path = os.path.join('./checkpoint',file_name)
record_path = os.path.join(check_path, file_name+'_record.txt')
resume = False
learning_rate = 0.1


test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

net = net.to(device)
net = torch.nn.DataParallel(net)  # parallel GPU
cudnn.benchmark = True

attacks_auto_method = ['apgd-ce', 'apgd-dlr', 'fab', 'square']

record_path_words(record_path, "Adversarial Attack Performance:\n")

#TODO: PGD Attack test
print("==> Loading best model:"+file_name+"\n")
assert os.path.isdir(check_path), 'Error: no checkpoint directory found!'
checkpoint_best = torch.load(os.path.join(check_path, 'model_best.pth.tar'))
checkpoint_last = torch.load(os.path.join(check_path, 'checkpoint.pth.tar'))
if config.Operation.Validate_Last == True:
    with open(record_path, "a+") as f:
        f.write("Last_trained_model Performance: \n")
    f.close()
    net.load_state_dict(checkpoint_last['state_dict'])
    ##----->PDG
    adversary_1 = LinfPGDAttack(net,loss_fn=nn.CrossEntropyLoss(),eps=eps_record/255.0,nb_iter=config.ADV.pgd_attack_1,eps_iter=step_record/255.0,
                              rand_init=True,clip_min=0.0,clip_max=1.0)
    adversary_2 = LinfPGDAttack(net,loss_fn=nn.CrossEntropyLoss(),eps=eps_record/255.0,nb_iter=config.ADV.pgd_attack_2,eps_iter=step_record/255.0,
                              rand_init=True,clip_min=0.0,clip_max=1.0)
    for adv_attack in [adversary_1, adversary_2]:
        test_acc_pgd = test_adv(net,adv_attack,test_loader)
        record_path_words(record_path, f"PGD_attack:[nb_iter:{adv_attack.nb_iter},eps:{eps_record},step_size:{step_record}]->pgd_acc: {test_acc_pgd: .2f}" + "\n")
    FGSM_attack = FGSM(net, loss_fn=nn.CrossEntropyLoss(), eps=eps_record/255)
    test_acc_attack = test_adv(net,FGSM_attack,test_loader)
    record_path_words(record_path,f"FGSM_attack:eps:{eps_record}->attack_acc: {test_acc_attack: .2f}" + "\n")
    AA_attack = AutoAttack(net, norm='Linf', eps=eps_record/255, attacks_to_run=attacks_auto_method, version='custom')
    test_acc_attack = test_adv_auto(net,AA_attack,test_loader)
    record_path_words(record_path,f"AA_attack:eps:{eps_record}->attack_acc: {test_acc_attack: .2f}" + "\n")

if config.Operation.Validate_Best == True:
    with open(record_path, "a+") as f:
        f.write("Best_model Performance: \n")
    f.close()
    net.load_state_dict(checkpoint_best['state_dict'])
    ##----->PDG
    adversary_1 = LinfPGDAttack(net,loss_fn=nn.CrossEntropyLoss(),eps=eps_record/255.0,nb_iter=config.ADV.pgd_attack_1,eps_iter=step_record/255.0,
                              rand_init=True,clip_min=0.0,clip_max=1.0)
    adversary_2 = LinfPGDAttack(net,loss_fn=nn.CrossEntropyLoss(),eps=eps_record/255.0,nb_iter=config.ADV.pgd_attack_2,eps_iter=step_record/255.0,
                              rand_init=True,clip_min=0.0,clip_max=1.0)
    for adv_attack in [adversary_1, adversary_2]:
        test_acc_pgd = test_adv(net,adv_attack,test_loader)
        record_path_words(record_path, f"PGD_attack:[nb_iter:{adv_attack.nb_iter},eps:{eps_record},step_size:{step_record}]->pgd_acc: {test_acc_pgd: .2f}" + "\n")
    #---->FGSM
    FGSM_attack = FGSM(net, loss_fn=nn.CrossEntropyLoss(), eps=eps_record/255 )
    test_acc_attack = test_adv(net,FGSM_attack,test_loader)
    record_path_words(record_path,f"FGSM_attack:eps:{eps_record}->attack_acc: {test_acc_attack: .2f}" + "\n")
    #---->AA
    AA_attack  = AutoAttack(net, norm='Linf', eps=eps_record/255, attacks_to_run=attacks_auto_method, version='custom')
    test_acc_attack = test_adv_auto(net,AA_attack,test_loader)
    record_path_words(record_path,f"AA_attack:eps:{eps_record}->attack_acc: {test_acc_attack: .2f}" + "\n")