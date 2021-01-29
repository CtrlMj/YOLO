import torch
import numpy as np
import torch.nn as nn
from collections import Counter

def convert_to_global_coords(predictions, split):
    
    batch_size = predictions.size(0)
    predictions = predictions.reshape(predictions.size(0), split, split, -1)
    bbox1 = predictions[..., 20:25]
    bbox2 = predictions[..., 25:30]
    confidence, maxIdx = torch.max(torch.stack([bbox1[..., 0:1], bbox2[..., 0:1]]), dim=0)
    bbox = (1 - maxIdx)*bbox1 + maxIdx*bbox2
    idx_mask = torch.arange(7).repeat((batch_size, S, 1)).to('cuda')
    bbox[..., 1] = (idx_mask + bbox[..., 1]) / S
    bbox[..., 2] = (idx_mask.permute(0, 2, 1) + bbox[..., 2]) / S
    bbox[..., 3] = bbox[..., 3] / 7
    bbox[..., 4] = bbox[..., 4] / 7
    objclass = torch.argmax(predictions[..., :20], dim=-1, keepdim=True).float()
    return torch.cat([objclass, bbox], dim=-1)

def listOfboxes(model, train_loader, iou_threshold, obj_threshold, split=7, device='cuda'):
    """
    predictions (batch_size, S*S*(n_classes + B*5))
    """
    model.eval()
    instance_id = 0
    unsuppressed_prediction_boxes = []
    target_boxes = []
    for batch_id, (x_train, targets) in enumerate(train_loader):
        with torch.no_grad():
            x_train, targets = x_train.to(device), targets.to(device)
            preds = model(x_train)
        batch_size = x_train.size(0)
        preds = convert_to_global_coords(preds, split)
        targets = convert_to_global_coords(targets, split)
        all_boxes = getBoxList(preds)
        all_tboxes = getBoxList(targets)
        
        for i in range(batch_size):
            unsuppresseds = non_max_suppression(all_boxes[i], iou_threshold, obj_threshold)
            for box in unsuppresseds:
                unsuppressed_prediction_boxes.append([instance_id] + box)
            for tbox in all_tboxes[i]:
                if tbox[1] == 1:
                    target_boxes.append([instance_id] + tbox)
            instance_id += 1
    
    model.train()
    return unsuppressed_prediction_boxes, target_boxes
def getBoxList(preds, split=7):
    batch_size = preds.size(0)
    preds = preds.reshape(batch_size, split * split, -1)
    lst = []
    for i in range(batch_size):
        ith_lst = []
        for j in range(split*split):
            box = preds[i, j, :]
            ith_lst.append(box.tolist())
        
        lst.append(ith_lst)
    
    return lst



    

def calculate_iou(box1, box2):
    """
    boxes are in shape = center_x, center_y, width, height
    box1 = (N, S, S, 4)
    box2 = (N, S, S, 4)
    """
    x11 = box1[...,0:1]-box1[...,2:3]/2
    y11 = box1[...,1:2]-box1[...,3:4]/2
    x12 = box1[...,0:1]+box1[...,2:3]/2
    y12 = box1[...,1:2]+box1[...,3:4]/2
    
    x21 = box2[...,0:1]-box2[...,2:3]/2
    y21 = box2[...,1:2]-box2[...,3:4]/2
    x22 = box2[...,0:1]+box2[...,2:3]/2
    y22 = box2[...,1:2]+box2[...,3:4]/2 
    
    x1, y1 = torch.max(x11, x21), torch.max(y11, y21)
    x2, y2 = torch.min(x12, x22), torch.min(y12, y22)
    
    union = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_s = box1[..., 2:3] * box1[..., 3:4]
    box2_s = box2[..., 2:3] * box2[..., 3:4]
    
    iou = union / (box1_s + box2_s - union + 1e-6)
    
    return iou



def mean_average_precision(pred_box, true_box, iou_threshold=0.5, num_classes=20):
    """
        pred_box = [[train_idx, class_pred, prob_score, x, y, width, height], ...]
        true_box = ``
    """
    average_precisions = []
    for c in range(num_classes):
        detections = []
        targets = []
        for prediction in pred_box:
            if prediction[1] == c:
                detections.append(prediction)
        for target in true_box:
            if target[1] == c:
                targets.append(target)
        if len(targets) == 0:
            continue
        
        detections.sort(key=lambda x: x[2], reverse=True)
        if len(detections) == 0:
            continue
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        
        img2bbx = Counter([target[0] for target in targets])
        img2bbx = {key: torch.zeros(value) for key, value in img2bbx.items()}
        
        for detection_idx, detection in enumerate(detections):
            best_iou = 0
            best_iou_idx = 0
            idx = 0
            for target in targets:
                if target[0] == detection[0]:
                    iou = calculate_iou(torch.tensor(target[3:]), torch.tensor(detection[3:]))
                    if iou > best_iou:
                        best_iou = iou
                        best_iou_idx = idx
                    idx += 1
            
            if best_iou > iou_threshold:
                if img2bbx[detection[0]][best_iou_idx] == 0:
                    img2bbx[detection[0]][best_iou_idx] = 1
                    TP[detection_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1
        
        TP_cumu = torch.cumsum(TP, dim=0)
        FP_cumu = torch.cumsum(FP, dim=0)
        recall = TP_cumu / (len(targets) + 1e-6)
        recall = torch.cat((torch.tensor([0.]), recall))
        precision = TP_cumu / (TP_cumu + FP_cumu + 1e-6)
        precision = torch.cat((torch.tensor([1.]), precision))
        
        average_precisions.append(torch.trapz(precision, recall))
    
    return sum(average_precisions) / (len(average_precisions) + 1e-6)

def non_max_suppression(predictions, iou_threshold, prob_threshold):
    """
    predictions = [[object_id, prob, x, y, width, height], ...]
    """
    predictions = [prediction for prediction in predictions if prediction[1] >= prob_threshold]
    suppressed_version = []
    predictions.sort(key = lambda x: x[1])
    while predictions:
        prediction = predictions.pop()
        
        predictions = [pred for pred in predictions if pred[0] != predictions[0] or calculate_iou(torch.tensor(pred[2:]), torch.tensor(prediction[2:])) < iou_threshold]
    
        suppressed_version.append(prediction)
    
    return suppressed_version

class transform():
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image):
        for transform in self.transforms:
            image = transform(image)
        
        return image
