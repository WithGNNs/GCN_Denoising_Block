from utils_AT import *
import torch
import sys
import numpy as np
import time
from torch.autograd import Variable

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def validate_pgd(val_loader, model, criterion, K, step, configs):
    # Mean/Std for normalization   
    mean = torch.Tensor(np.array(configs.TRAIN.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3,configs.DATA.crop_size, configs.DATA.crop_size).cuda()
    std = torch.Tensor(np.array(configs.TRAIN.std)[:, np.newaxis, np.newaxis])
    std = std.expand(3, configs.DATA.crop_size, configs.DATA.crop_size).cuda()
    # Initiate the meters
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    eps = configs.ADV.clip_eps
    step /= configs.DATA.max_color_value

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        orig_input = input.clone()
        randn = torch.FloatTensor(input.size()).uniform_(-eps, eps).cuda()
        input += randn
        input.clamp_(0, 1.0)
        for _ in range(K):
            invar = Variable(input, requires_grad=True)
            in1 = invar - mean
            in1.div_(std)
            output = model(in1)
            ascend_loss = criterion(output, target)
            ascend_grad = torch.autograd.grad(ascend_loss, invar)[0]
            pert = fgsm(ascend_grad, step)
            # Apply purturbation
            input += pert.data
            input = torch.max(orig_input-eps, input)
            input = torch.min(orig_input+eps, input)
            input.clamp_(0, 1.0)
        
        input.sub_(mean).div_(std)
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % configs.TRAIN.print_freq == 0:
                print('PGD Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
                sys.stdout.flush()

    print(' PGD Final Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
        .format(top1=top1, top5=top5))

    return top1.avg

def validate(val_loader, model, criterion, configs, logger=None):
    # Mean/Std for normalization   
    mean = torch.Tensor(np.array(configs.TRAIN.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3,configs.DATA.crop_size, configs.DATA.crop_size).cuda()
    std = torch.Tensor(np.array(configs.TRAIN.std)[:, np.newaxis, np.newaxis])
    std = std.expand(3, configs.DATA.crop_size, configs.DATA.crop_size).cuda()
    
    # Initiate the meters
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            input = input - mean
            input.div_(std)
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % configs.TRAIN.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
                sys.stdout.flush()

    print(' Final Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))
    return top1.avg

def validate_fgsm(val_loader, model, criterion, eps, configs):
    # Mean/Std for normalization
    mean = torch.Tensor(np.array(configs.TRAIN.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3, configs.DATA.crop_size, configs.DATA.crop_size).cuda()
    std = torch.Tensor(np.array(configs.TRAIN.std)[:, np.newaxis, np.newaxis])
    std = std.expand(3, configs.DATA.crop_size, configs.DATA.crop_size).cuda()

    # Initiate the meters
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # Set epsilon
    eps /= configs.DATA.max_color_value

    # Evaluation mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # Set requires_grad attribute of tensor for backprop
        input.requires_grad = True
        # Forward pass
        output = model(input)
        # Calculate loss
        loss = criterion(output, target)
        # Calculate gradients
        model.zero_grad()
        loss.backward()
        # Collect data for statistics
        with torch.no_grad():
            # Get gradient sign
            sign_data_grad = input.grad.sign()
            # Perturb the input
            perturbed_input = input + eps * sign_data_grad
            # Re-normalize input to keep the pixel values within [0,1] range
            perturbed_input = torch.clamp(perturbed_input, min=0, max=1)
            # Normalize the input
            perturbed_input = (perturbed_input - mean) / std
            # Evaluate perturbed input
            output = model(perturbed_input)
            loss = criterion(output, target)
            # Measure accuracy and record loss
            prec1, _ = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % configs.TRAIN.print_freq == 0:
                print(
                    "FGSM Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                        i,
                        len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                    )
                )
                sys.stdout.flush()

    print("FGSM Final Prec@1 {top1.avg:.3f}".format(top1=top1))
    return top1.avg

from autoattack import AutoAttack

def validate_autoattack(test_loader, model, eps, configs):
    # Mean/Std for normalization
    mean = torch.Tensor(np.array(configs.TRAIN.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3, configs.DATA.crop_size, configs.DATA.crop_size).cuda()
    std = torch.Tensor(np.array(configs.TRAIN.std)[:, np.newaxis, np.newaxis])
    std = std.expand(3, configs.DATA.crop_size, configs.DATA.crop_size).cuda()

    eps /= configs.DATA.max_color_value

    # Evaluation mode
    model.eval()
    test_images = []
    test_labels = []

    for images, labels in test_loader:
        test_images.append(images)
        test_labels.append(labels)

    test_images = torch.cat(test_images, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    adversary = AutoAttack(model, norm='Linf', eps=eps, attacks_to_run=['apgd-ce', 'apgd-dlr', 'fab', 'square'], version='custom')
    inputs, targets = test_images.to(device), test_labels.to(device)
    adv_complete = adversary.run_standard_evaluation((inputs- mean) / std, targets, bs=250)
    
    adv_outputs = []
    total = len(adv_complete)
    batch_size = 100
    for i in range(0, total, batch_size):
        adv_batch = adv_complete[i:i+batch_size]
        adv_outputs_batch = model(adv_batch)
        adv_outputs.append(adv_outputs_batch.cpu().detach().numpy())
    adv_output = np.concatenate(adv_outputs, axis=0)

    # Measure accuracy and record loss
    _, predicted = torch.max(torch.from_numpy(adv_output), 1)
    predicted = predicted.to(device)
    adv_correct = predicted.eq(targets).sum().item()
    acc = adv_correct/total

    print(f"AutoAttack Final Prec@1 {acc:.3f}")

    return adv_correct

def record_path_words(record_path, record_words):
    print(record_words)
    with open(record_path, "a+") as f:
        f.write(record_words)
    f.close()
    return
