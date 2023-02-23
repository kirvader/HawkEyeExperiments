import sys
from pathlib import Path

import cv2
import torch
from numpy import random

project_root = Path(__file__).parent.parent.parent

sys.path.append(str(project_root))

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, \
    scale_coords, xyxy2xywh, set_logging, increment_path
from utils.torch_utils import select_device, time_synchronized, TracedModel
from experiments.inference_utils.detection_result import DetectionResult


class Args:
    def __init__(self,
                 weights='yolov7.pt',
                 source='inference/images',
                 img_size=640,
                 conf_thres=0.25,
                 iou_thres=0.45,
                 device='',
                 view_img=True,
                 save_txt=True,
                 save_conf=False,
                 nosave=True,
                 classes=None,
                 agnostic_nms=False,
                 augment=False,
                 update=False,
                 project='runs/detect',
                 name='exp',
                 exist_ok=True,
                 no_trace=False):
        if classes is None:
            classes = [0, 32]
        self.weights = weights
        self.source = source
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.view_img = view_img
        self.save_txt = save_txt
        self.save_conf = save_conf
        self.nosave = nosave
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.update = update
        self.project = project
        self.name = name
        self.exist_ok = exist_ok
        self.no_trace = no_trace


class YOLOv7SingleDetectionRunner:
    def __init__(self, opt: Args):
        source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
        self.augment = opt.augment
        self.conf_thres = opt.conf_thres
        self.iou_thres = opt.iou_thres
        self.classes = opt.classes
        self.agnostic_nms = opt.agnostic_nms

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        self.device = select_device(opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        self.img_sz = imgsz

        if trace:
            self.model = TracedModel(self.model, self.device, opt.img_size)

        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        if self.device.type != 'cpu':
            self.model(
                torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.old_img_w = self.old_img_h = imgsz
        self.old_img_b = 1

    def run(self, original_frame):
        resized_frame, ratio, (dw, dh) = letterbox(original_frame, self.img_sz)
        resized_frame = resized_frame[:, :, ::-1].transpose(2, 0, 1).copy()
        resized_frame = torch.from_numpy(resized_frame).to(self.device)
        resized_frame = resized_frame.half() if self.half else resized_frame.float()  # uint8 to fp16/32
        resized_frame /= 255.0  # 0 - 255 to 0.0 - 1.0
        if resized_frame.ndimension() == 3:
            resized_frame = resized_frame.unsqueeze(0)

        # Warmup
        if self.device.type != 'cpu' and (
                self.old_img_b != resized_frame.shape[0] or self.old_img_h != resized_frame.shape[
            2] or self.old_img_w != resized_frame.shape[3]):
            self.old_img_b = resized_frame.shape[0]
            self.old_img_h = resized_frame.shape[2]
            self.old_img_w = resized_frame.shape[3]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = self.model(resized_frame, augment=self.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes,
                                   agnostic=self.agnostic_nms)
        t3 = time_synchronized()

        result = []
        # Process detections
        det = pred[0]  # single frame
        s = ""

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(resized_frame.shape[2:], det[:, :4], original_frame.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

            gn = torch.tensor(original_frame.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # Write results
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                result.append(DetectionResult(xywh[0], xywh[1], xywh[2], xywh[3], cls, conf))

        # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
        return result


if __name__ == "__main__":
    detector640 = YOLOv7SingleDetectionRunner(Args())
    image = cv2.imread(f"{str(project_root)}/inference/images/image2.jpg")
    print(detector640.run(image))
