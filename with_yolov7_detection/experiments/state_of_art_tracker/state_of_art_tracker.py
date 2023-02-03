import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent

sys.path.append(str(project_root))

from experiments.inference_utils.detection_result import Box, DetectionResult
from experiments.inference_utils.single_frame_yolov7_detector import YOLOv7SingleDetectionRunner, Args
from experiments.run_single_hypothesis import run_solution
from experiments.tracker_base import SingleObjectTrackerBase


class StateOfArtDetector(SingleObjectTrackerBase):
    def __init__(self, inference_time: int = 350, tracking_cls: int = 32):
        self.detector = YOLOv7SingleDetectionRunner(Args(weights="yolov7-e6e.pt", classes=[tracking_cls]))
        self.detector_inference_time = inference_time
        self.tracking_cls = tracking_cls

    def is_available(self, timestamp: int) -> bool:
        return True

    def process_frame(self, frame, timestamp: int) -> Box:
        results = self.detector.run(frame)
        if len(results) == 0:
            return None

        return max(results, key=lambda detection: detection.conf).box

    def get_prediction_area(self, timestamp: int) -> Box:
        return Box(0.5, 0.5, 1.0, 1.0)


if __name__ == "__main__":
    tracker = StateOfArtDetector()
    run_solution(tracker, "inference/1.mp4", "inference/1_output.mp4", "inference/1_raw.json")
