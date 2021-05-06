#Coding=utf-8

import os
import cv2
import math
import random
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from IPython import embed

class Trainer:

    def __init__(self, config, model, train_dataloader, val_dataloader, optimizer, criterion):

        self.device = config['device']
        self.dir_checkpoint = config['dir_checkpoint']

        self.batch_size = config['batch_size']
        self.n_epoch = config['n_epoch']

        self.best_loss = math.inf

        self.model = model.to(self.device)
        print('[Log] Model loaded sucessfully\n')
            
        self.optimizer = optimizer
        self.criterion = criterion

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        print('[Log] Datasets loaded | Training : %d | Validation : %d\n'%(len(self.train_dataloader.dataset), len(self.val_dataloader.dataset)))

    def train_epoch(self, epoch):
        
        self.model.train()

        n = len(self.train_dataloader)

        epoch_loss = 0

        for i, data in enumerate(self.train_dataloader):
    
            data = {k : v.to(self.device) for k, v in data.items()}
            input = data['input']
            truth = data['truth']

            self.optimizer.zero_grad()
            pred = self.model(input)

            loss = self.criterion(pred, truth)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss

            print('[Epoch %04d] %d / %d'%(epoch, i + 1, n), end='\r')
        
        epoch_loss = epoch_loss / n
        print('[Epoch %04d] Loss : %.4f'%(epoch, epoch_loss))

        return epoch_loss

    def val_epoch(self, epoch):

        self.model.eval()
        
        n = len(self.val_dataloader)

        epoch_loss = 0

        with torch.no_grad():

            for i, data in enumerate(self.val_dataloader):
                    
                data = {k : v.to(self.device) for k, v in data.items()}
                input = data['input']
                truth = data['truth']

                pred = self.model(input)
           
                loss = self.criterion(pred, truth)

                epoch_loss += loss

                print('[Validating] %d / %d'%(i + 1, n), end='\r')

            epoch_loss = epoch_loss / n
            print('[Validation] Loss : %.4f'%(epoch_loss))

        return epoch_loss

    def train(self):

        for epoch in range(self.n_epoch):
            train_loss = self.train_epoch(epoch)
            val_loss = self.val_epoch(epoch)
            
            if (val_loss < self.best_loss):
                self.best_loss = val_loss
                Path(self.dir_checkpoint).mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch' : epoch,
                    'loss' : self.best_loss,
                    'state_dict' : self.model.state_dict()
                }, os.path.join(self.dir_checkpoint, 'best_model.pth'))    
                print('[Log] Saving Checkpoint ...'%(self.best_loss))
            print('\n')

