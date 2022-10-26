import torch
import cv2
from torchvision import transforms
import torch.nn as nn
import numpy as np
import random

import sys, os

from zmq import device
sys.path.append("/home/sophie/yolov7")

from utils.general import non_max_suppression, scale_coords, set_logging, check_img_size
from utils.plots import plot_one_box
from utils.torch_utils import TracedModel, select_device
from models.common import Conv
from utils.datasets import letterbox, LoadImages
from models.experimental import attempt_load

with torch.no_grad():
    source = "file.jpg"
    weights = "experiments/yolov7.pt"
    imsize = 640
    trace = True
    augment = False

    set_logging()
    device = select_device('')
    half = device.type != 'cpu'

    model = attempt_load(weights,map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imsize, s=stride)

    if trace:
        model = TracedModel(model, device, imsize)

    if half:
        model.half()

    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=augment)[0]

        with torch.no_grad():
            pred = model(img, augment=augment)[0]
        
        pred = non_max_suppression(pred)
        print(pred)

