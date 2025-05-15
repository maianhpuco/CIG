import torch
import numpy as np
import torch
from utils.utils import *
import os


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict



def train_one_epoch(features, label, optimizer, criterion, milnet, args):
    
    epoch_loss = 0
    for i, data in enumerate(bag_ins_list):
        optimizer.zero_grad()
        
        
        data_tensor = torch.from_numpy(np.stack(data_bag_list)).float().cuda()
        data_tensor = data_tensor[:, 0:args.num_feats]
        label = torch.from_numpy(np.array(int(np.clip(data[0], 0, 1)))).float().cuda()
        
        classes, bag_prediction, _, _ = milnet(data_tensor) # n X L
        
        max_prediction, index = torch.max(classes, 0)
        
        loss_bag = criterion(bag_prediction.view(1, -1), label.view(1, -1))
        loss_max = criterion(max_prediction.view(1, -1), label.view(1, -1))
        loss_total = 0.5*loss_bag + 0.5*loss_max
        loss_total = loss_total.mean()
        
        loss_total.backward()
        optimizer.step()  
        
        epoch_loss = epoch_loss + loss_total.item()
    
    return epoch_loss / len(bag_ins_list)
