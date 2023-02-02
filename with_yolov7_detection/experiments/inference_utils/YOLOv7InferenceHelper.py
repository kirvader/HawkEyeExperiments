import sys
from pathlib import Path

import cv2

sys.path.append(Path(__file__).parent.parent.parent.__str__())

from single_frame_yolov7_detector import YOLOv7SingleDetectionRunner, Args


