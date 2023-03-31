import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent

sys.path.append(str(project_root))

from experiments.inference_utils.detection_result import Box, DetectionResult, transform_to_absolute_from_relative
from experiments.inference_utils.single_frame_yolov7_detector import YOLOv7SingleDetectionRunner, Args
from experiments.run_single_hypothesis import run_solution
from experiments.tracker_base import SingleObjectTrackerBase


class YOLOv7OnlyDetectionTracker(SingleObjectTrackerBase):
    def __init__(self, inference_time: int = 350, tracking_cls: int = 32, successful_detection_relevance_time: int = 1000):
        self.last_inference_start_time = 0
        self.detector = None  # setup is required
        self.detector_inference_time = inference_time
        self.tracking_cls = tracking_cls
        self.weights = "yolov7.pt"
        self.last_successful_detection_data = (None, 0)  # Box and timestamp
        self.prelast_successful_detection_data = (None, 0)  # Box and timestamp
        self.successful_detection_relevance_time = successful_detection_relevance_time
        self.scale_is_natural = False

    def setup(self, filename: str):
        with open(filename, "r") as f:
            json_object = json.load(f)
        self.tracking_cls = json_object["tracking_cls"]
        self.detector_inference_time = json_object["detector_inference_time"]
        self.weights = json_object["weights"]
        self.successful_detection_relevance_time = json_object["successful_detection_relevance_time"]

        self.detector = YOLOv7SingleDetectionRunner(Args(classes=[self.tracking_cls], weights=self.weights))
        self.scale_is_natural = bool(json_object["successful_detection_relevance_time"])

    def export_config(self, filename: str):
        with open(filename, "w") as f:
            f.write(json.dumps({
                "tracking_cls": self.tracking_cls,
                "detector_inference_time": self.detector_inference_time,
                "weights": self.weights,
                "successful_detection_relevance_time": self.successful_detection_relevance_time,
                "scale_is_natural": self.scale_is_natural
            }, indent=4))

    def is_available(self, timestamp: int) -> bool:
        return timestamp - self.last_inference_start_time >= self.detector_inference_time

    def process_frame(self, frame, timestamp: int) -> Box:
        self.last_inference_start_time = timestamp
        results = []
        if not self.scale_is_natural:
            results = self.detector.run(frame)
        else: # only if the size of model is less than all the frame dimensions
            height = frame.shape[0]
            width = frame.shape[1]
            model_size = 640

            for top_index in range(0, height, model_size):
                bottom_index = min(top_index + model_size, height)
                calibrated_top_index = bottom_index - model_size
                for left_index in range(0, width, model_size):
                    right_index = min(left_index + model_size, width)
                    calibrated_left_index = right_index - model_size
                    current_slice = frame[calibrated_top_index:bottom_index, calibrated_left_index:right_index]
                    current_slice_results = self.detector.run(current_slice)
                    container_box = Box((calibrated_left_index + right_index) / 2 / width,
                                        (calibrated_top_index + bottom_index) / 2 / height,
                                        (right_index - calibrated_left_index) / width,
                                        (bottom_index - calibrated_top_index) / height)
                    for current_slice_result in current_slice_results:
                        current_slice_result.box = transform_to_absolute_from_relative(current_slice_result.box, container_box)
                    results += current_slice_results
        self.prelast_successful_detection_data = self.last_successful_detection_data
        if len(results) == 0:
            return None

        result = max(results, key=lambda detection: detection.conf).box
        self.last_successful_detection_data = (result, timestamp)

        return result

    def get_estimate_position(self, timestamp: int) -> Box:
        if timestamp - self.last_successful_detection_data[1] > self.successful_detection_relevance_time:
            self.last_successful_detection_data = (None, self.last_successful_detection_data[1])
        return self.last_successful_detection_data[0]

    def get_real_time_estimate_position(self, timestamp: int) -> Box:
        if timestamp - self.prelast_successful_detection_data[1] > self.successful_detection_relevance_time:
            self.prelast_successful_detection_data = (None, self.prelast_successful_detection_data[1])
        return self.prelast_successful_detection_data[0]

    def get_prediction_area(self, timestamp: int) -> Box:
        return Box(0.5, 0.5, 1.0, 1.0)


if __name__ == "__main__":
    tracker = YOLOv7OnlyDetectionTracker()
    tracker.export_config("inference/configs/pure_yolov7_detector/default.json")
    # run_solution(tracker, "inference/1.mp4", "inference/1_raw.json")
