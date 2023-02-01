import os
import sys
from pathlib import Path

import torch


sys.path.append(Path(__file__).parent.parent.__str__())

from utils.general import strip_optimizer
from detect import detect

class Args:
    def __init__(self):
        self.weights = 'yolov7.pt'
        self.source = 'inference/images'
        self.img_size = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.device = ''
        self.view_img = True
        self.save_txt = True
        self.save_conf = False
        self.nosave = True
        self.classes = [32]
        self.agnostic_nms = False
        self.augment = False
        self.update = False
        self.project = 'runs/detect'
        self.name = 'exp'
        self.exist_ok = True
        self.no_trace = False

opt = Args()

with torch.no_grad():
    if opt.update:  # update all models (to fix SourceChangeWarning)
        for opt.weights in ['yolov7.pt']:
            detect(opt)
            strip_optimizer(opt.weights)
    else:
        detect(opt)
