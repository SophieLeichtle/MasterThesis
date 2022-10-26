import torch
import cv2
from torchvision import transforms
import torch.nn as nn
import numpy as np
import random

import sys, os
sys.path.append("/home/sophie/yolov7")

from utils.general import non_max_suppression, scale_coords, set_logging, check_img_size
from utils.plots import plot_one_box
from utils.torch_utils import TracedModel, select_device
from models.common import Conv
from utils.datasets import letterbox, LoadImages
from models.experimental import attempt_load


class YoloHelper:
    @torch.no_grad()
    def __init__(self, 
    weights='experiments/yolov7.pt',
    imgsz = 640,
    trace = True,
    save_img = True,
    augment = False,
    conf_thresh = 0.25,
    iou_thresh = 0.45,
    agnostic_nms = False):
        self.trace = trace
        self.save_img = save_img
        self.augment = augment
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.agnostic_nms = agnostic_nms
        set_logging()
        self.device = select_device('')
        self.half = self.device.type != 'cpu'
        
        self.model = attempt_load(weights, map_location=self.device)

        self.stride = int(self.model.stride.max())
        self.imgsz = check_img_size(imgsz, s=self.stride)

        if trace:
            self.model = TracedModel(self.model,self.device,imgsz)

        if self.half: 
            self.model.half()

        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0,255) for _ in range(3)] for _ in self.names]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.old_img_w = self.old_img_h = imgsz
        self.old_img_b = 1
        dataset = LoadImages("dog.jpg", img_size=640, stride=self.stride)
        for path, img, im0, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()
            img /= 255.0
            im0s = im0
        #img = np.ascontiguousarray(img.transpose(2,0,1))
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
            self.old_img_b = img.shape[0]
            self.old_img_h = img.shape[2]
            self.old_img_w = img.shape[3]
            for i in range(3):
                self.model(img, augment=self.augment)[0]

        with torch.no_grad():
            pred = self.model(img,augment=self.augment)[0]
        print(pred[...,4])
        pred = non_max_suppression(pred, self.conf_thresh, self.iou_thresh, self.agnostic_nms)
        print(pred)

    def getDetections(self, image):
        with torch.no_grad():
            self.model.to(self.device)
            dataset = LoadImages("dog.jpg", img_size=640, stride=self.stride)
            for path, img, im0, vid_cap in dataset:
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()
                img /= 255.0
                im0s = im0
            #img = np.ascontiguousarray(img.transpose(2,0,1))
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
                self.old_img_b = img.shape[0]
                self.old_img_h = img.shape[2]
                self.old_img_w = img.shape[3]
                for i in range(3):
                    self.model(img, augment=self.augment)[0]

            with torch.no_grad():
                pred = self.model(img,augment=self.augment)[0]
            print(pred[...,4])
            pred = non_max_suppression(pred, self.conf_thresh, self.iou_thresh, self.agnostic_nms)
            print(pred)
            for i, det in enumerate(pred):
                im0 = im0s
                s = ''

                if len(det):
                    det[:,:4] = scale_coords(img.shape[2:], det[:,:4], im0.shape).round()
                    
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    print(s)
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy,im0,label=label, color = self.colors[int(cls)], line_thickness=1)

                if self.save_img:
                    cv2.imwrite("seg.png", im0)
            
        return im0