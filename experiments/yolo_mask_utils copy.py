import torch
import cv2
import yaml
import numpy as np

import sys
sys.path.append("/home/sophie/yolov7")

from utils.datasets import letterbox
from utils.general import xywh2xyxy, box_iou

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image

import torchvision
import time
from torch.nn import functional as F

@torch.no_grad()
def create_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open('experiments/hyp.scratch.mask.yaml') as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)

    weights = torch.load('experiments/yolov7-mask.pt')
    model = weights['model']
    model.half().to(device)
    _ = model.eval()

    return model, device, hyp

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
def get_predictions(img_tensor, model, hyp):
    with torch.no_grad():
        output = model(img_tensor)
    names = model.names
    inf_out, train_out, attn, mask_iou, bases, sem_output = output['test'], output['bbox_and_cls'], output['attn'], output['mask_iou'], output['bases'], output['sem']
    bases = torch.cat([bases, sem_output], dim=1)
    pooler_scale = model.pooler_scale
    pooler = ROIPooler(output_size=hyp['mask_resolution'], scales=(pooler_scale,), sampling_ratio=1, pooler_type='ROIAlignV2', canonical_level=2)
    output, output_mask, output_mask_score, output_ac, output_ab = non_max_suppression_mask_conf(inf_out, attn, bases, pooler, hyp, conf_thres=0.25, iou_thres=0.65, merge=False, mask_iou=None)
    pred, pred_masks = output[0], output_mask[0]
    return pred, pred_masks, names

@torch.no_grad()
def save_seg_image(path, img_tensor, pred, pred_masks, hyp, names):
    bboxes = Boxes(pred[:,:4])
    
    nb, _, height, width = img_tensor.shape
    original_pred_masks = pred_masks.view(-1, hyp['mask_resolution'], hyp['mask_resolution'])
    pred_masks = retry_if_cuda_oom(paste_masks_in_image)(original_pred_masks, bboxes, (height, width), threshold=0.5)
    pred_masks_np = pred_masks.detach().cpu().numpy()
    pred_cls = pred[:, 5].detach().cpu().numpy()
    pred_conf = pred[:, 4].detach().cpu().numpy()
    nimg = img_tensor[0].permute(1,2,0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    nbboxes = bboxes.tensor.detach().cpu().numpy().astype(np.int)
    pnimg = nimg.copy()

    for one_mask, bbox, cls, conf in zip(pred_masks_np, nbboxes, pred_cls, pred_conf):
        if conf < 0.25: continue
        color = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]
        pnimg[one_mask] = pnimg[one_mask] * 0.5 + np.array(color, dtype=np.uint8) * 0.5
        pnimg = cv2.rectangle(pnimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        label = '%s %.3f' % (names[int(cls)], conf)
        t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
        c2 = bbox[0] + t_size[0], bbox[1] - t_size[1] - 3
        pnimg = cv2.rectangle(pnimg, (bbox[0], bbox[1]), c2, color, -1, cv2.LINE_AA)  # filled
        pnimg = cv2.putText(pnimg, label, (bbox[0], bbox[1] - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA) 
        cv2.imwrite(path, pnimg)

def get_detection(env, model, device, hyp, label, save_image = False):
    state = env.get_state()
    img = state["rgb"]
    img = np.clip(img*255, 0, 255)
    img = img.astype(np.uint8)
    img_tensor, brg_img = prepare_image(img[:,:,:3], device)
    predictions, pred_masks, names = get_predictions(img_tensor, model, hyp)
    print(predictions)
    if save_image:
        save_seg_image("insseg.png", img_tensor, predictions, pred_masks, hyp, names)
    pred_cls = predictions[:, 5].detach().cpu().numpy()
    pred_conf = predictions[:, 4].detach().cpu().numpy()
    index = -1
    for i in range(len(pred_cls)):
        if pred_conf[i] < 0.25: continue
        if names[int(pred_cls[i])] == label:
            index = i
            break
    if index == -1: return None, None

    bboxes = Boxes(predictions[index,:4])
    bbox = Boxes.tensor.detach().cpu().numpy().astype(np.int)
    bbox = bbox[0,:]
    nb, _, height, width = img_tensor.shape
    original_pred_masks = pred_masks.view(-1, hyp['mask_resolution'], hyp['mask_resolution'])
    pred_mask = retry_if_cuda_oom(paste_masks_in_image)(original_pred_masks[index,:,:], bboxes, (height,width), threshold=0.5)
    pred_mask_np = pred_mask.detach().cpu().numpy()
    pred_mask_np = pred_mask_np[0, :, :]

    robot_pos = env.robots[0].get_position()[:2]
    robot_theta = env.robots[0].get_rpy()[2]

    center_col = (int(bbox[2]) + int(bbox[0])) / 2
    f = 579.4
    theta_rel = np.arctan2(320 - center_col,f)
    detection_theta = robot_theta + theta_rel
    newest_detection = [robot_pos[0], robot_pos[1], detection_theta]
    return newest_detection, pred_mask_np


def merge_bases(rois, coeffs, attn_r, num_b, location_to_inds=None):
    # merge predictions
    # N = coeffs.size(0)
    if location_to_inds is not None:
        rois = rois[location_to_inds]
    N, B, H, W = rois.size()
    if coeffs.dim() != 4:
        coeffs = coeffs.view(N, num_b, attn_r, attn_r)
    # NA = coeffs.shape[1] //  B
    coeffs = F.interpolate(coeffs, (H, W),
                           mode="bilinear").softmax(dim=1)
    # coeffs = coeffs.view(N, -1, B, H, W)
    # rois = rois[:, None, ...].repeat(1, NA, 1, 1, 1)
    # masks_preds, _ = (rois * coeffs).sum(dim=2) # c.max(dim=1)
    masks_preds = (rois * coeffs).sum(dim=1)
    return masks_preds

def non_max_suppression_mask_conf(prediction, attn, bases, pooler, hyp, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False, mask_iou=None, vote=False):

    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32
    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    output_mask = [None] * prediction.shape[0]
    output_mask_score = [None] * prediction.shape[0]
    output_ac = [None] * prediction.shape[0]
    output_ab = [None] * prediction.shape[0]
    
    def RMS_contrast(masks):
        mu = torch.mean(masks, dim=-1, keepdim=True)
        return torch.sqrt(torch.mean((masks - mu)**2, dim=-1, keepdim=True))
    
    
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        
        # If none remain process next image
        if not x.shape[0]:
            continue
            
        a = attn[xi][xc[xi]]
        base = bases[xi]

        bboxes = Boxes(box)
        pooled_bases = pooler([base[None]], [bboxes])
        
        pred_masks = merge_bases(pooled_bases, a, hyp["attn_resolution"], hyp["num_base"]).view(a.shape[0], -1).sigmoid()

        if mask_iou is not None:
            mask_score = mask_iou[xi][xc[xi]][..., None]
        else:
            temp = pred_masks.clone()
            temp[temp < 0.5] = 1 - temp[temp < 0.5]
            mask_score = torch.exp(torch.log(temp).mean(dim=-1, keepdims=True))#torch.mean(temp, dim=-1, keepdims=True)
        
        x[:, 5:] *= x[:, 4:5] * mask_score # x[:, 4:5] *   * mask_conf * non_mask_conf  # conf = obj_conf * cls_conf

        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            mask_score = mask_score[i]
            if attn is not None:    
                pred_masks = pred_masks[i]
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]


        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue
        
        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        # scores *= mask_score
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
            
        
        all_candidates = []
        all_boxes = []
        if vote:
            ious = box_iou(boxes[i], boxes) > iou_thres
            for iou in ious: 
                selected_masks = pred_masks[iou]
                k = min(10, selected_masks.shape[0])
                _, tfive = torch.topk(scores[iou], k)
                all_candidates.append(pred_masks[iou][tfive])
                all_boxes.append(x[iou, :4][tfive])
        #exit()
            
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        output_mask_score[xi] = mask_score[i]
        output_ac[xi] = all_candidates
        output_ab[xi] = all_boxes
        if attn is not None:
            output_mask[xi] = pred_masks[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output, output_mask, output_mask_score, output_ac, output_ab