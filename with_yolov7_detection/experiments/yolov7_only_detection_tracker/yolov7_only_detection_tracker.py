from experiments.inference_utils.detection_result import Box
from experiments.inference_utils.timer import Timer
from experiments.tracker_base import SingleObjectTrackerBase


class YOLOv7OnlyDetectionTracker(SingleObjectTrackerBase):
    def __init__(self):
        self.timer = Timer()

    def is_available(self) -> bool:
        return self.timer.is_expired()

    def process_frame(self, frame, timestamp: int) -> Box:
        pass

    def get_prediction_area(self, timestamp: int) -> Box:
        return Box(0.5, 0.5, 1.0, 1.0)


