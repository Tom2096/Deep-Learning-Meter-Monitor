#Coding=utf-8

import os
import cv2
import json
import torch
import numpy as np

from PIL import Image
from torchvision import transforms

from pathlib import Path

from IPython import embed

resize_width = 512
resize_height = 512

def save_heatmaps(image, heatmaps, filename):
    
    Path('heatmaps').mkdir(parents=True, exist_ok=True)
    basename = os.path.splitext(filename)[0]
    
    image = np.array(image)
    image = cv2.resize(image, (resize_width // 4, resize_height // 4))
    cv2.imwrite(os.path.join('heatmaps', filename), image)

    for i, heatmap in enumerate(heatmaps):
        
        hm = heatmap.copy()
        hm = (hm * 255).astype(np.uint8)
        hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)

        cv2.imwrite(os.path.join('heatmaps', basename + '_%d.jpg'%(i)), hm)

class Dataset:

    def __init__(self, task='train', shuffle=False):

        if (task == 'train'):
            self.root = os.path.join('dataset', 'train') 
        elif (task == 'val'):
            self.root = os.path.join('dataset', 'val')
        else:
            raise NotImplementedError
        
        self.list_dir = os.listdir(os.path.join(self.root, 'images'))

        with open(os.path.join(self.root, 'labels.txt'), 'r', encoding='utf-8') as txt_file:
            
            self.labels = {}
            
            for label in txt_file.readlines():

                label = label.replace('\n', '')
                label = label.split(' ')
                
                assert (len(label) == 7) # Name, x1, y1, x2, y2, x3, y3

                filename = label[0]
                coords = label[1:]

                self.labels[filename] = torch.tensor(list(map(float, coords)))
            
        if (shuffle):
            random.shuffle(self.list_dir)
        
        self.preprocess = transforms.Compose([
            transforms.Resize((resize_height, resize_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.heatmap_stride = 4
        self.heatmap_sigma = 3

    def get_heatmaps(self, size, label):

        heatmap_width = size[1] // self.heatmap_stride
        heatmap_height = size[0] // self.heatmap_stride

        points = [
            (int(label[0] * heatmap_width), int(label[1] * heatmap_height)),
            (int(label[2] * heatmap_width), int(label[3] * heatmap_height)),
            (int(label[4] * heatmap_width), int(label[5] * heatmap_height))
        ]

        heatmaps = np.zeros((3, heatmap_height, heatmap_width), dtype=np.float32)

        tmp_size = self.heatmap_sigma * 3

        for i, point in enumerate(points):

            heatmap_size = 2 * tmp_size + 1
            x = np.arange(0, heatmap_size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = heatmap_size // 2

            # Focal Map for One Point #######################################
                
            focal = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.heatmap_sigma ** 2))

            # Paste on Heatmap ################################################

            top_left = (max(0, point[0] - tmp_size), max(0, point[1] - tmp_size))
            bot_right = (min(heatmap_width, point[0] + tmp_size + 1), min(heatmap_height, point[1] + tmp_size + 1))

            start_x = max(0, -(point[0] - tmp_size))
            end_x = min(focal.shape[1], (heatmap_width + focal.shape[1]) - (point[0] + tmp_size + 1))

            start_y = max(0, -(point[1] - tmp_size))
            end_y = min(focal.shape[0], (heatmap_height + focal.shape[0]) - (point[1] + tmp_size + 1))
        
            heatmaps[i, top_left[1]:bot_right[1], top_left[0]:bot_right[0]] = focal[start_y:end_y, start_x:end_x]

        return heatmaps

    def __getitem__(self, idx):
        
        filename = self.list_dir[idx]

        image = Image.open(os.path.join(self.root, 'images', filename))
        input = image.copy()
        input = self.preprocess(input)
        label = self.labels[filename]
        heatmaps = self.get_heatmaps(input.shape[1:], label)

        #save_heatmaps(image, heatmaps, filename)

        return {
            'input' : input,
            'truth' : heatmaps
        }

    def __len__(self):

        return len(self.list_dir)

