import torch
import math
import numpy as np
from scipy.fftpack import fft, fftfreq
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv,GATv2Conv
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.utils import negative_sampling,get_laplacian,to_undirected,get_laplacian


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Re_Graph(nn.Module):
    def __init__(self,feature_map) -> None:
        super(Re_Graph,self).__init__()
        self.feature_map = feature_map
        self.GAP = torch.nn.functional.adaptive_avg_pool2d(feature_map, (1,1))
        self.batch_size = feature_map.shape[0]
        self.num_feature_map = feature_map.shape[1]
        self.image_size = feature_map.shape[2]
        self.graph_feature = feature_map.view(feature_map.shape[0],feature_map.shape[1],feature_map.shape[2]**2)
        #TODO：GCN fliter
        self.conv_fliter = GCNConv(feature_map.shape[2]**2, feature_map.shape[2]**2)

    def forward(self, k:int=5, undirected:bool=True):
        feature_after_fliter = torch.zeros_like(self.feature_map)
        #TODO：Compute Similarity Matrix
        S = self.get_euclidean()
        all_graph_edge = self.reconstruct_graph(S,k).type(torch.long)
        for i in range(self.batch_size):
            if undirected == True:
                data = Data(x=self.graph_feature[i], edge_index=to_undirected(all_graph_edge[i].T)).to(device)
            else:
                data = Data(x=self.graph_feature[i], edge_index=all_graph_edge[i].T).to(device)
            feature_after_fliter[i] = torch.relu(self.conv_fliter(data.x, data.edge_index)).view(self.num_feature_map,self.image_size,self.image_size)
        return feature_after_fliter+self.feature_map
    
    def calculate_feature_map_similarity(self, input_tensor):
        num_feature_map =  input_tensor.shape[1]
    
        feature_map_reshaped = torch.reshape(input_tensor, (self.batch_size, num_feature_map, -1))
        feature_map_norm = torch.norm(feature_map_reshaped, dim=2, keepdim=True)
        feature_map_dot_prod = torch.bmm(feature_map_reshaped, feature_map_reshaped.transpose(1, 2))
        similarity_matrix = feature_map_dot_prod / (feature_map_norm * feature_map_norm.transpose(1, 2))
        similarity_matrix = similarity_matrix.masked_fill(torch.eye(num_feature_map, device=input_tensor.device).bool(), -math.inf)
    
        return similarity_matrix

    def get_euclidean(self):
        nodes = self.GAP  # S = torch.ger(v, v)
        Similarity = torch.zeros((len(nodes), nodes.shape[1], nodes.shape[1]))
        for i in range(len(nodes)):
            X = nodes[i].squeeze()
            diff = X.view(-1,1) - X.view(1,-1)
            Similarity[i] = (-torch.pow(diff,2)).fill_diagonal_(-float('inf'))
        return Similarity

    def reconstruct_graph(self, Similarity, k):
        """
        Similarity: shape->(batch_size,num_feature_map,num_feature_map)
        k: top_k
        return: shape->[batch_size,num_edges,2]
        """
        edge_matrix = torch.zeros(self.batch_size,k*self.num_feature_map,2)
        #连边的索引-----节点2
        node_indices = torch.topk(Similarity, k, dim=2, largest=True, sorted=True, out=None)[1] #shape: [batch_size,num_feature_map,k]
        node_indices_flat = node_indices.view(self.batch_size,-1)
        #连边的索引-----节点1
        edge_indices = torch.arange(self.num_feature_map).repeat_interleave(k)
        for i in range(len(edge_matrix)):
            edge_matrix[i][:,0] = edge_indices
            edge_matrix[i][:,1] = node_indices_flat[i]
        self.edge_matrix = edge_matrix
        return edge_matrix

class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNet_GNN(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_GNN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, k=5, un_directed = True):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv_forward(out,k,un_directed)
        out = self.layer1(out)
        # out = self.conv_forward(out,k,un_directed)
        out = self.layer2(out)
        # out = self.conv_forward(out,k,un_directed)
        out = self.layer3(out)
        # out = self.conv_forward(out,k,un_directed)
        out = self.layer4(out)
        # out = self.conv_forward(out, k, un_directed)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def conv_forward(self,x,k,un_directed):
        graph = Re_Graph(x).to(device)
        return graph.forward(k,un_directed)
    
class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out

class WideResNet(nn.Module):
    def __init__(self, num_classes, depth, widen_factor, dropout_rate=0.0):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        n = int((depth - 4) / 6)
        k = widen_factor

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._wide_layer(WideBasicBlock, 16*k, n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasicBlock, 32*k, n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasicBlock, 64*k, n, dropout_rate, stride=2)
        self.bn = nn.BatchNorm2d(64*k)
        self.linear = nn.Linear(64*k, num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class DotProductNonLocalMeans(nn.Module):
    def __init__(self, in_channels):
        super(DotProductNonLocalMeans, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        f = torch.matmul(x.view(batch_size, channels, -1).permute(0, 2, 1), x.view(batch_size, channels, -1))
        f_div_C = f / (height * width)
        y = torch.matmul(x.view(batch_size, channels, -1), f_div_C).view(batch_size, channels, height, width)
        y = self.conv1x1(y)
        return y
    
class WideResNet_GNN(nn.Module):
    def __init__(self, num_classes, depth, widen_factor, block=[1],dropout_rate=0.0):
        super(WideResNet_GNN, self).__init__()
        self.in_planes = 16
        self.block = block

        n = int((depth - 4) / 6)
        k = widen_factor

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._wide_layer(WideBasicBlock, 16*k, n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasicBlock, 32*k, n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasicBlock, 64*k, n, dropout_rate, stride=2)
        self.bn = nn.BatchNorm2d(64*k)
        self.linear = nn.Linear(64*k, num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def forward(self, x, k=5, un_directed = True):
        out = F.relu(self.conv1(x))
        if 1 in self.block:
            out = self.conv_forward(out,k,un_directed)
        out = self.layer1(out)
        if 2 in self.block:
            out = self.conv_forward(out,k,un_directed)
        out = self.layer2(out)
        if 3 in self.block:
            out = self.conv_forward(out,k,un_directed)
        out = self.layer3(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def conv_forward(self,x,k,un_directed):
        graph = Re_Graph(x).to(device)
        return graph.forward(k,un_directed)


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

def ResNet18_GNN():
    return ResNet_GNN(BasicBlock, [2,2,2,2])

def WRN32_10_GNN(block_pos):
    return WideResNet_GNN(num_classes=10, depth=32, widen_factor=10, block=block_pos)

def WRN32_10():
    return WideResNet(num_classes=10, depth=32, widen_factor=10)
