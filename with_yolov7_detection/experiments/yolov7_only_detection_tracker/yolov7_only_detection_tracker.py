import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent

sys.path.append(str(project_root))

from experiments.inference_utils.detection_result import Box, DetectionResult
from experiments.inference_utils.single_frame_yolov7_detector import YOLOv7SingleDetectionRunner, Args
from experiments.run_single_hypothesis import run_solution
from experiments.tracker_base import SingleObjectTrackerBase


class YOLOv7OnlyDetectionTracker(SingleObjectTrackerBase):
    def __init__(self, inference_time: int = 350, tracking_cls: int = 32):
        self.last_inference_start_time = 0
        self.detector = YOLOv7SingleDetectionRunner(Args(classes=[tracking_cls]))
        self.detector_inference_time = inference_time
        self.tracking_cls = tracking_cls
        self.weights = "yolov7.pt"

    def setup(self, filename: str):
        with open(filename, "r") as f:
            json_object = json.load(f)
        self.tracking_cls = json_object["tracking_cls"]
        self.detector_inference_time = json_object["detector_inference_time"]
        self.weights = json_object["weights"]

        self.detector = YOLOv7SingleDetectionRunner(Args(classes=[self.tracking_cls], weights=self.weights))

    def export_config(self, filename: str):
        with open(filename, "w") as f:
            f.write(json.dumps({
                "tracking_cls": self.tracking_cls,
                "detector_inference_time": self.detector_inference_time,
                "weights": self.weights
            }, indent=4))

    def is_available(self, timestamp: int) -> bool:
        return timestamp - self.last_inference_start_time >= self.detector_inference_time

    def process_frame(self, frame, timestamp: int) -> Box:
        self.last_inference_start_time = timestamp
        results = self.detector.run(frame)
        if len(results) == 0:
            return None

        return max(results, key=lambda detection: detection.conf).box

    def get_prediction_area(self, timestamp: int) -> Box:
        return Box(0.5, 0.5, 1.0, 1.0)


if __name__ == "__main__":
    tracker = YOLOv7OnlyDetectionTracker()
    tracker.export_config("inference/configs/pure_yolov7_detector/default.json")
    # run_solution(tracker, "inference/1.mp4", "inference/1_raw.json")
