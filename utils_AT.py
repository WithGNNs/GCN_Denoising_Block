import logging
import os
import datetime
import torchvision.models as models
import math
import torch
import yaml
import numpy as np
from easydict import EasyDict
import shutil

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(initial_lr, optimizer, epoch, n_repeats):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (0.1 ** (epoch // int(math.ceil(30./n_repeats))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def fgsm(gradz, step_size):
    return step_size*torch.sign(gradz)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def initiate_logger(output_path):
    if not os.path.isdir(os.path.join('output', output_path)):
        os.makedirs(os.path.join('output', output_path))
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(os.path.join('output', output_path, 'log.txt'),'w'))
    logger.info(pad_str(' LOGISTICS '))
    logger.info('Experiment Date: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))
    logger.info('Output Name: {}'.format(output_path))
    return logger

def get_model_names():
	return sorted(name for name in models.__dict__
    		if name.islower() and not name.startswith("__")
    		and callable(models.__dict__[name]))

def pad_str(msg, total_len=70):
    rem_len = total_len - len(msg)
    return '*'*int(rem_len/2) + msg + '*'*int(rem_len/2)\

def parse_config_file(args):
    with open(args.config) as f:
        config = EasyDict(yaml.load(f))
        
    # Add args parameters to the dict
    for k, v in vars(args).items():
        config[k] = v
        
    # Add the output path
    config.output_name = '{:s}_step{:d}_eps{:d}_repeat{:d}'.format(args.output_prefix,
                         int(config.ADV.fgsm_step), int(config.ADV.clip_eps), 
                         config.ADV.n_repeats)
    return config


def save_checkpoint(state, is_best, filepath):
    filename = os.path.join(filepath, 'checkpoint.pth.tar')
    # Save model
    torch.save(state, filename)
    # Save best model
    if is_best:
        shutil.copyfile(filename, os.path.join(filepath, 'model_best.pth.tar'))

from torch.nn import functional as F
def evaluate_standard(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss / n, test_acc / n

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

    # 将列表中的数据合并为一个大的张量或数组
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