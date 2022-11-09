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

@torch.no_grad()
def create_model(weights, image_size, trace):
    device = select_device('')
    half = device.type != 'cpu'

    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(image_size, s=stride)

    if trace:
        model = TracedModel(model, device, image_size)
    if half:
        model.half()

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0,255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            
    return model, device, names, colors
        
@torch.no_grad()
def get_predictions(img_tensor, model):
    with torch.no_grad():
        pred = model(img_tensor)[0]
    pred = non_max_suppression(pred)
    return pred

@torch.no_grad()
def prepare_image(image, device, img_size=640, stride=32):
    half = device.type != 'cpu'
    
    brg_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    yolo_img = letterbox(image, img_size, stride)[0]
    yolo_img = yolo_img.transpose(2,0,1)
    yolo_img = np.ascontiguousarray(yolo_img)

    img_tensor = torch.from_numpy(yolo_img).to(device)
    img_tensor = img_tensor.half() if half else img_tensor.float()
    img_tensor /= 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    return img_tensor, brg_img

@torch.no_grad()
def save_seg_image(path, pred, img_tensor, brg_img, names, colors):
    im0 = brg_img
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:,:4], im0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
        cv2.imwrite(path, im0)

