#Coding=utf-8

import os
import cv2
import math
import json
import torch
import imutils
import torch.backends.cudnn as cudnn
import random
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from shapely.geometry import Point, LineString

from models.yolact import Yolact
from layers.output_utils import postprocess
from utils.augmentations import FastBaseTransform
from data import cfg

from models.customnet import CustomNet

from IPython import embed

############################## Helper Functions ###############################

def warp_image(image, points): # points must be in [top_left, top_right, bottom_right, bottom_left]
    
    warped = image.copy()
    top_left, top_right, bottom_right, bottom_left = points
    
    width_top = np.sqrt((top_left[0] - top_right[0]) ** 2 + (top_left[1] - top_right[1]) ** 2)
    width_bottom = np.sqrt((bottom_left[0] - bottom_right[0]) ** 2 + (bottom_left[1] - bottom_right[1]) ** 2)
    max_width = int(max(width_top, width_bottom)) 

    height_left = np.sqrt((top_left[0] - bottom_left[0]) ** 2 + (top_left[1] - bottom_left[1]) ** 2)
    height_right = np.sqrt((top_right[0] - bottom_right[0]) ** 2 + (top_right[1] - bottom_right[1]) ** 2)
    max_height = int(max(height_left, height_right)) 

    src = np.array([
        top_left,
        top_right,
        bottom_right,
        bottom_left
    ]).astype(np.float32)

    dst = np.array([
        (0, 0),
        (max_width - 1, 0),
        (max_width - 1, max_height - 1),
        (0, max_height - 1)
    ]).astype(np.float32)

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(warped, matrix, (max_width, max_height))

    return warped

############################## Model Classes ##################################

class HR_Net:

    def __init__(self, pth_checkpoint):
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        checkpoint = torch.load(pth_checkpoint, map_location=self.device)
        self.net = CustomNet().to(self.device)
        self.net.load_state_dict(checkpoint['state_dict'])
        self.net.eval()
        
        self.resize_width = 512 
        self.resize_height = 512

        self.preprocess = transforms.Compose([
            transforms.Resize((self.resize_height, self.resize_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def extrapolate(self, line):
        
        ratio = 10000
      
        p1 = list(line.coords)[0]
        p2 = list(line.coords)[1]

        p1 = tuple([
            p1[0] - (ratio * (p2[0] - p1[0])),
            p1[1] - (ratio * (p2[1] - p1[1]))    
        ])

        p2 = tuple([
            p2[0] + (ratio * (p2[0] - p1[0])),
            p2[1] + (ratio * (p2[1] - p1[1]))
        ])
        
        return LineString([p1, p2])

    def form_line_from_slope(self, point, slope):

        point_a = (point.x, point.y)
        point_b = (point.x + 1, point.y + slope)
        line = LineString([point_a, point_b])
        line = self.extrapolate(line)
        
        return line

    def get_slope(self, line):

        point_a = list(line.coords)[0]
        point_b = list(line.coords)[1]

        slope = (point_b[1] - point_a[1]) / (point_b[0] - point_a[0]) 

        return slope

    def reflect_point_about(self, point, about):

        x = point.x - about.x
        y = point.y - about.y

        reflected = Point([(about.x - x), (about.y - y)])
        
        return reflected

    def get_prediction(self, heatmaps):
        
        batch_size = heatmaps.shape[0]
        num_joints = heatmaps.shape[1]

        heatmap_height = heatmaps.shape[2]
        heatmap_width = heatmaps.shape[3]

        reshaped = heatmaps.reshape((batch_size, num_joints, -1))
        idx = np.argmax(reshaped, axis=2)
        maxvals = np.amax(reshaped, axis=2)

        maxvals = maxvals.reshape((batch_size, num_joints, 1))
        idx = idx.reshape((batch_size, num_joints, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

        preds[:, :, 0] = (preds[:, :, 0]) % heatmap_width
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / heatmap_width)

        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)

        preds *= pred_mask
        preds = preds.squeeze(0)
        preds[:, 0] /= heatmap_width
        preds[:, 1] /= heatmap_height

        return preds, maxvals

    def construct_bbx(self, predictions):

        left_point = Point(predictions[0])
        right_point = Point(predictions[2])
        
        bottom_point = Point(predictions[1])
        top_point = self.reflect_point_about(bottom_point, Point([
            (left_point.x + right_point.x) // 2, 
            (left_point.y + right_point.y) // 2
        ]))

        cross_line = LineString([left_point, right_point])
        slope = self.get_slope(cross_line)
        perp_slope = -(1 / slope)

        left_line = self.form_line_from_slope(left_point, perp_slope)
        right_line = self.form_line_from_slope(right_point, perp_slope)
        bottom_line = self.form_line_from_slope(bottom_point, slope)
        top_line = self.form_line_from_slope(top_point, slope)

        top_left = left_line.intersection(top_line)
        top_right = right_line.intersection(top_line)
        bottom_left = left_line.intersection(bottom_line)
        bottom_right = right_line.intersection(bottom_line)

        return [
            tuple(map(int, list(top_left.coords)[0])),
            tuple(map(int, list(top_right.coords)[0])),
            tuple(map(int, list(bottom_right.coords)[0])),
            tuple(map(int, list(bottom_left.coords)[0]))
        ]

    def infer(self, image):
        
        with torch.no_grad():

            input = image.copy()
            input = self.preprocess(input).unsqueeze(0).to(self.device)
            heatmaps = np.array(self.net(input).cpu())

            predictions, confidence = self.get_prediction(heatmaps)
            
            predictions[:, 0] *= image.width
            predictions[:, 1] *= image.height
            
            predictions = predictions.astype(np.int)
            bbx = self.construct_bbx(predictions)
            assert (len(bbx) == 4)

            warped = warp_image(np.array(image), bbx)

            return warped

class Yolact_Net:

    def __init__(self, pth_checkpoint):

        if torch.cuda.is_available():
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.device = torch.device('cuda')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
            self.device = torch.device('cpu')

        self.net = Yolact()
        self.net.load_weights(pth_checkpoint)
        self.net.eval()
        self.net = self.net.to(self.device)

        self.fast_nms = True
        self.cross_class_nms = False
        self.mask_proto_debug = False

        self.net.detect.use_fast_nms = True
        self.net.use_cross_class_nms = False
        cfg.mask_proto_debug = False

        self.display_lincomb = False
        self.crop = True

        self.top_k = 15
        self.score_threshold = 0.15
    
    def read_pointer(self, mask):
        
        cords = mask.nonzero()[::-1]
        cords = np.vstack(cords)
        middle = cords.mean(axis=1).reshape(2, 1)
        cords = cords-middle

        cov = np.cov(cords)
        ei_val, ei_vec = np.linalg.eig(cov)
        ei_val = ei_val.real
        ei_vec = ei_vec.real
        
        idx = ei_val.argmax()
        axis = ei_vec[:, idx]
        projections = cords.T.dot(axis)
        
        axis = axis if (np.median(projections) < 0) else -axis
        
        axis[1] *= -1

        x, y = axis
        
        if ((x < 0) and (y > 0)):
            angle = 450 - np.arctan2(y, x) * 180 / np.pi        
        else:
            angle = 90 - np.arctan2(y, x) * 180 / np.pi
        
        return angle / 360

    def order_points(self, pts):

        top_left, top_right = sorted(sorted(pts, key=lambda p : p[1])[:2], key=lambda p : p[0])
        bottom_left, bottom_right = sorted(sorted(pts, key=lambda p : p[1])[2:], key=lambda p : p[0])

        average_x = sum([p[0] for p in pts]) / 4 
        average_y = sum([p[1] for p in pts]) / 4
        
        if (top_left[0] < average_x and bottom_left[0] < average_x):
            if (top_right[0] > average_x and bottom_right[0] > average_x):
                if (top_left[1] < average_y and top_right[1] < average_y):
                    if (bottom_left[1] > average_y and bottom_right[1] > average_y):
                        return [top_left, top_right, bottom_right, bottom_left]
        
        return None

    def read_meter(self, image, mask):
        
        blur = cv2.GaussianBlur(mask, (5, 5), 0)
        edges = cv2.Canny(blur, 100, 200)
        
        contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        hull = cv2.convexHull(contour)

        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * peri, True).squeeze(1)
        approx = [tuple([p[0], p[1]]) for p in approx]
        
        return None

    def infer(self, image):

        with torch.no_grad():

            frame = torch.from_numpy(image).float().to(self.device)
            batch = FastBaseTransform()(frame.unsqueeze(0))
            preds = self.net(batch)

            img_gpu = frame / 255.0
            h, w, _ = frame.shape
            
            save = cfg.rescore_bbox
            cfg.rescore_bbox = True
            t = postprocess(preds, w, h, visualize_lincomb = self.display_lincomb,
                                            crop_masks        = self.crop,
                                            score_threshold   = self.score_threshold)
            cfg.rescore_bbox = save

            idx = t[1].argsort(0, descending=True)[:self.top_k]
            
            if cfg.eval_mask_branch:
                # Masks are drawn on the GPU, so don't copy
                mask = t[3][idx]
            classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

            num_dets_to_consider = min(self.top_k, classes.shape[0])
            for j in range(num_dets_to_consider):
                if scores[j] < self.score_threshold:
                    num_dets_to_consider = j
                    break
            
            pointer_mask = None
            meter_mask = None

            if (num_dets_to_consider > 0):

                mask = mask[:num_dets_to_consider, :, :, None]
                
                for i, m in enumerate(mask):

                    m = np.array(m.cpu()) * 255
                    m = m.squeeze(2).astype(np.uint8)
                    
                    if (classes[i] == 0):
                        pointer_mask = m if (pointer_mask is None) else None
                    else:
                        meter_mask = m if (meter_mask is None) else None

            if ((pointer_mask is not None) and (meter_mask is not None)):

                pointer_reading = self.read_pointer(pointer_mask)
                meter_reading = self.read_meter(image, meter_mask)

                return pointer_mask, pointer_reading

###############################################################################

def main(config):

    hr_net = HR_Net(config['hr_net'])
    yolact = Yolact_Net(config['yolact'])

    with torch.no_grad():
        
        for filename in os.listdir(config['test_dir']):
            
            basename = os.path.splitext(filename)[0]
            
            image = Image.open(os.path.join(config['test_dir'], filename))

            warped = hr_net.infer(image)
            pointer_mask, pointer_reading = yolact.infer(warped)

            ########################### Matplotlib Visualization ###########################

            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20,10))
            
            axs[0].imshow(image)
            axs[0].set_title('Input', fontsize=20)
            axs[0].axis('off')

            axs[1].imshow(warped)
            axs[1].set_title('Warped', fontsize=20)
            axs[1].axis('off')

            axs[2].imshow(pointer_mask)
            axs[2].set_title('Mask | Reading = %.2f'%(pointer_reading), fontsize=20)
            axs[2].axis('off')
            
            Path('results').mkdir(parents=True, exist_ok=True)
            fig.suptitle('Pipeline', fontsize=50)
            fig.savefig(os.path.join('results', basename + '.jpg'))

if __name__ == '__main__':  

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    random.seed(0)
    
    config = {

        'device' : torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        'hr_net' : './pretrained/hr_net.pth',
        'yolact' : './pretrained/yolact.pth',

        'test_dir' : 'test'
    }

    main(config)

    
































