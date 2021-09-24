from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch
import cv2
import time
import numpy as np


def plot_bboxes(image, bboxes, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        if cls_id in ['smoke', 'phone', 'eat']:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        if cls_id == 'eat':
            cls_id = 'eat-drink'
        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, '{} ID-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return image


def update_tracker(dets, image, deepsort):
    if len(dets):
        xywhs = torch.rand(dets.shape[0], 4)
        xywhs[:, 0] = torch.from_numpy(((dets[:, 0] + dets[:, 2]) / 2).round())
        xywhs[:, 1] = torch.from_numpy(((dets[:, 1] + dets[:, 3]) / 2).round())
        xywhs[:, 2] = torch.from_numpy(dets[:, 2] - dets[:, 0])
        xywhs[:, 3] = torch.from_numpy(dets[:, 3] - dets[:, 1])
        # clss = torch.rand(dets.shape[0], )
        # clss = dets[:, 5].clone()
        clss = torch.from_numpy(dets[:, 5])
        # Pass detections to deepsort
        # t1 = time.time()
        outputs = deepsort.update(xywhs, clss, image)
        # print("total time", time.time() - t1)
    else:
        outputs = np.array([[0., 0., 0., 0., 0., 0.]])
    return outputs
