#Coding=utf-8

import os
import json
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.customnet import CustomNet
from models.loss import JointsMSELoss
from utils.dataset import Dataset
from utils.trainer import Trainer

from IPython import embed

def main(config):

    model = CustomNet()
    print('[Log] Preparing training\n')

    train_dataset = Dataset(task='train')
    val_dataset = Dataset(task='val')

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'])
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = JointsMSELoss(use_target_weight=False)

    trainer = Trainer(config, model, train_dataloader, val_dataloader, optimizer, criterion)
    trainer.train()

if __name__ == "__main__":  

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    random.seed(0)
    
    config = {

        'device' : torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        'lr' : 0.001,
        'n_epoch' : 100,
        'batch_size' : 4,
        'dir_checkpoint' : './checkpoints/w_512_h_512_e_100'
    }

    main(config)

    