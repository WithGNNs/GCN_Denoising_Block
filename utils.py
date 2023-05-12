import math
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm, tqdm_notebook
import csv
import numpy as np
import os
import scipy.sparse as sp
import scipy.stats as st

from random import triangular
from typing import Callable, Union
import os
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Set, Tuple

istype = lambda x, y: x.type(y.dtype)

device = torch.device("cuda:0")
trn = transforms.Compose([transforms.ToTensor(),])

## simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    def forward(self, x):
        #[None,:,None,None] - >make (3) array extend to (1,3,1,1)
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]

class Dataset_ImageNet(torch.utils.data.Dataset):
    def __init__(self, image, label):
        self.image = image
        self.label = label
    
    def __getitem__(self, item):
        return self.image[item], self.label[item]
        
    def __len__(self):
        return len(self.image)

class Accumulator:
    def __init__(self,n):
        self.data = [0.0]*n  

    def add(self, *args):
        self.data = [a+float(b) for a,b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def format_time(seconds):
    """
    cur_time = time.time()
    time.sleep(64)
    last_time = time.time()
    step_time = format_time(last_time-cur_time)
    print(step_time)
    """
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    return f


def load_ImageNet_iter(Dataset_Image_net, batch_size):
    """
    for i, (X, y) in enumerate(train_iter):
    """
    iter = torch.utils.data.DataLoader(dataset=Dataset_Image_net, batch_size=batch_size, shuffle=True)
    return iter

##load image metadata (Image_ID, true label, and target label)
def load_ground_truth(csv_filename):
    image_id_list = []
    label_ori_list = []
    label_tar_list = []

    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            image_id_list.append( row['ImageId'] )
            label_ori_list.append( int(row['TrueLabel']) - 1 )
            label_tar_list.append( int(row['TargetClass']) - 1 )

    return image_id_list,label_ori_list,label_tar_list

def load_imagenet1k(input_path,image_id_list,label_list,img_size=299):
    X = torch.zeros(1000,3,img_size,img_size).to(device)
    for i in range(1000):          
        X[i] = trn(Image.open(input_path + image_id_list[i] + '.png'))  
    labels = torch.tensor(label_list).to(device)
    return X, labels

def accuracy_imagenet1k(model,input_path,image_id_list,label_list,batch_size=20,num_batches=50,img_size=299):
    acc = 0
    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    for k in range(0,num_batches):
        batch_size_cur = min(batch_size,len(image_id_list) - k * batch_size)  #目前的batch_size大小，batch_size或者最后一轮剩余的数量    
        X_ori = torch.zeros(batch_size_cur,3,img_size,img_size).to(device)  #初始化image特征
        for i in range(batch_size_cur):          
            X_ori[i] = trn(Image.open(input_path + image_id_list[k * batch_size + i] + '.png'))  
        labels = torch.tensor(label_list[k * batch_size:k * batch_size + batch_size_cur]).to(device)   
        acc += sum(torch.argmax(model(norm(X_ori)),dim=1) == labels).cpu().numpy()
    print("Accuracy:"+str(acc/(batch_size*num_batches)))
    return acc/(batch_size*num_batches)

def Get_edge_pyg(A):
  edge_index_temp = sp.coo_matrix(A)
  values = edge_index_temp.data  # 对应权重值weight
  indices = np.vstack((edge_index_temp.row, edge_index_temp.col))  
  # edge_index_A = torch.LongTensor(indices)  
  edge = torch.LongTensor(indices) 
  edge_attribute = torch.FloatTensor(values) 
  edge_index = torch.sparse_coo_tensor(edge, edge_attribute, edge_index_temp.shape)
  return edge_index, edge, edge_attribute

def load_dataset(
        dataset: Dataset,
        n_examples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = 100
    loader = data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=0)
    x_test, y_test = [], []
    for i, (x, y) in enumerate(loader):
        x_test.append(x)
        y_test.append(y)
        if n_examples is not None and batch_size * i >= n_examples:
            break
    x_test_tensor = torch.cat(x_test)
    y_test_tensor = torch.cat(y_test)

    if n_examples is not None:
        x_test_tensor = x_test_tensor[:n_examples]
        y_test_tensor = y_test_tensor[:n_examples]

    return x_test_tensor, y_test_tensor

def imshow(img, title):
    img = torchvision.utils.make_grid(img.cpu().data, normalize=True)
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.show()


