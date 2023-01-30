import torch
import cv2
import yaml
import numpy as np

import sys

sys.path.append("/home/sophie/yolov7")

from utils.datasets import letterbox
from utils.general import xywh2xyxy, box_iou
from utils.general import non_max_suppression_mask_conf

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image

import torchvision
import time
from torch.nn import functional as F
import os


@torch.no_grad()
def create_model(directory="yolo_files"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(directory, "hyp.scratch.mask.yaml")) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)

    weights = torch.load(os.path.join(directory, "yolov7-mask.pt"))
    model = weights["model"]
    model.half().to(device)
    _ = model.eval()

    return model, device, hyp


@torch.no_grad()
def prepare_image(image, device, img_size=640, stride=32):
    half = device.type != "cpu"

    brg_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    yolo_img, ratio, padding = letterbox(image, img_size, stride)
    yolo_img = yolo_img.transpose(2, 0, 1)
    yolo_img = np.ascontiguousarray(yolo_img)

    img_tensor = torch.from_numpy(yolo_img).to(device)
    img_tensor = img_tensor.half() if half else img_tensor.float()
    img_tensor /= 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    return img_tensor, brg_img, padding


@torch.no_grad()
def get_predictions(img_tensor, model, hyp):
    with torch.no_grad():
        output = model(img_tensor)
    names = model.names
    inf_out, train_out, attn, mask_iou, bases, sem_output = (
        output["test"],
        output["bbox_and_cls"],
        output["attn"],
        output["mask_iou"],
        output["bases"],
        output["sem"],
    )
    bases = torch.cat([bases, sem_output], dim=1)
    pooler_scale = model.pooler_scale
    pooler = ROIPooler(
        output_size=hyp["mask_resolution"],
        scales=(pooler_scale,),
        sampling_ratio=1,
        pooler_type="ROIAlignV2",
        canonical_level=2,
    )
    (
        output,
        output_mask,
        output_mask_score,
        output_ac,
        output_ab,
    ) = non_max_suppression_mask_conf(
        inf_out,
        attn,
        bases,
        pooler,
        hyp,
        conf_thres=0.25,
        iou_thres=0.65,
        merge=False,
        mask_iou=None,
    )
    pred, pred_masks = output[0], output_mask[0]
    return pred, pred_masks, names


@torch.no_grad()
def save_seg_image(path, img_tensor, pred, pred_masks, hyp, names):
    bboxes = Boxes(pred[:, :4])

    nb, _, height, width = img_tensor.shape
    original_pred_masks = pred_masks.view(
        -1, hyp["mask_resolution"], hyp["mask_resolution"]
    )
    pred_masks = retry_if_cuda_oom(paste_masks_in_image)(
        original_pred_masks, bboxes, (height, width), threshold=0.5
    )
    pred_masks_np = pred_masks.detach().cpu().numpy()
    pred_cls = pred[:, 5].detach().cpu().numpy()
    pred_conf = pred[:, 4].detach().cpu().numpy()
    nimg = img_tensor[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    nbboxes = bboxes.tensor.detach().cpu().numpy().astype(np.int)
    pnimg = nimg.copy()

    for one_mask, bbox, cls, conf in zip(pred_masks_np, nbboxes, pred_cls, pred_conf):
        if conf < 0.25:
            continue
        color = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]
        pnimg[one_mask] = pnimg[one_mask] * 0.5 + np.array(color, dtype=np.uint8) * 0.5
        pnimg = cv2.rectangle(pnimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        label = "%s %.3f" % (names[int(cls)], conf)
        t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
        c2 = bbox[0] + t_size[0], bbox[1] - t_size[1] - 3
        pnimg = cv2.rectangle(
            pnimg, (bbox[0], bbox[1]), c2, color, -1, cv2.LINE_AA
        )  # filled
        pnimg = cv2.putText(
            pnimg,
            label,
            (bbox[0], bbox[1] - 2),
            0,
            0.5,
            [255, 255, 255],
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        cv2.imwrite(path, pnimg)


def get_detection(env, model, device, hyp, label, save_image=False):
    state = env.get_state()
    img = state["rgb"]
    img = np.clip(img * 255, 0, 255)
    img = img.astype(np.uint8)
    img_tensor, brg_img, padding = prepare_image(img[:, :, :3], device)
    predictions, pred_masks, names = get_predictions(img_tensor, model, hyp)
    if predictions is None:
        return None, None
    if save_image:
        save_seg_image("insseg.png", img_tensor, predictions, pred_masks, hyp, names)
    pred_cls = predictions[:, 5].detach().cpu().numpy()
    pred_conf = predictions[:, 4].detach().cpu().numpy()
    index = -1
    for i in range(len(pred_cls)):
        if pred_conf[i] < 0.6:
            continue
        if names[int(pred_cls[i])] == label:
            index = i
            break
    if index == -1:
        return None, None

    bboxes = Boxes(predictions[:, :4])
    bbox = bboxes.tensor.detach().cpu().numpy().astype(np.int)
    bbox = bbox[index, :]
    nb, _, height, width = img_tensor.shape
    original_pred_masks = pred_masks.view(
        -1, hyp["mask_resolution"], hyp["mask_resolution"]
    )
    pred_mask = retry_if_cuda_oom(paste_masks_in_image)(
        original_pred_masks, bboxes, (height, width), threshold=0.5
    )
    pred_mask_np = pred_mask.detach().cpu().numpy()
    pred_mask_np = pred_mask_np[index, :, :]

    robot_pos = env.robots[0].get_position()[:2]
    robot_theta = env.robots[0].get_rpy()[2]

    center_col = (int(bbox[2]) + int(bbox[0])) / 2
    f = 579.4
    theta_rel = np.arctan2(320 - center_col, f)
    detection_theta = robot_theta + theta_rel
    newest_detection = [robot_pos[0], robot_pos[1], detection_theta]

    return newest_detection, pred_mask_np


def get_detections(env, model, device, hyp, label, save_image=False):
    state = env.get_state()
    img = state["rgb"]
    img = np.clip(img * 255, 0, 255)
    img = img.astype(np.uint8)
    img_tensor, brg_img, padding = prepare_image(img[:, :, :3], device)
    predictions, pred_masks, names = get_predictions(img_tensor, model, hyp)
    if predictions is None:
        return None, None
    if save_image:
        save_seg_image("insseg.png", img_tensor, predictions, pred_masks, hyp, names)
    pred_cls = predictions[:, 5].detach().cpu().numpy()
    pred_conf = predictions[:, 4].detach().cpu().numpy()
    indexes = []
    for i in range(len(pred_cls)):
        if pred_conf[i] < 0.6:
            continue
        if names[int(pred_cls[i])] == label:
            indexes.append(i)
    if len(indexes) == 0:
        return None, None

    detections = []
    masks = []

    for index in indexes:
        bboxes = Boxes(predictions[:, :4])
        bbox = bboxes.tensor.detach().cpu().numpy().astype(np.int)
        bbox = bbox[index, :]
        nb, _, height, width = img_tensor.shape
        original_pred_masks = pred_masks.view(
            -1, hyp["mask_resolution"], hyp["mask_resolution"]
        )
        pred_mask = retry_if_cuda_oom(paste_masks_in_image)(
            original_pred_masks, bboxes, (height, width), threshold=0.5
        )
        pred_mask_np = pred_mask.detach().cpu().numpy()
        pred_mask_np = pred_mask_np[index, :, :]

        robot_pos = env.robots[0].get_position()[:2]
        robot_theta = env.robots[0].get_rpy()[2]

        center_col = (int(bbox[2]) + int(bbox[0])) / 2
        f = 579.4
        theta_rel = np.arctan2(320 - center_col, f)
        detection_theta = robot_theta + theta_rel
        newest_detection = [robot_pos[0], robot_pos[1], detection_theta]

        detections.append(newest_detection)
        masks.append(pred_mask_np)

    return detections, masks
