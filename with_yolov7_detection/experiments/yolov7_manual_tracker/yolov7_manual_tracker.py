import sys
from math import log, exp
from pathlib import Path

project_root = Path(__file__).parent.parent.parent

sys.path.append(str(project_root))

from experiments.inference_utils.detection_result import Box, DetectionResult, transform_to_absolute_from_relative
from experiments.inference_utils.single_frame_yolov7_detector import YOLOv7SingleDetectionRunner, Args
from experiments.run_single_hypothesis import run_solution
from experiments.tracker_base import SingleObjectTrackerBase
from experiments.yolov7_manual_tracker.manual_trackers_utils import get_cropped_image


class YOLOv7ManualTracker(SingleObjectTrackerBase):
    def __init__(self, tracking_cls: int = 32):
        self.last_good_result_timestamp = -10000
        self.last_good_result = Box(0.5, 0.5, 1.0, 1.0)

        self.center_velocity = (0.0, 0.0)
        self.prediction_area_size_velocity = (1.0, 1.0)
        self.max_delta_time = 600
        self.velocity_coef = 0.5 #  mini deceleration

        self.next_inference_timestamp = 0
        self.detectors = [(YOLOv7SingleDetectionRunner(Args(classes=[tracking_cls], img_size=512)), 350, 0.8),
                          (YOLOv7SingleDetectionRunner(Args(classes=[tracking_cls], img_size=256)), 250, 0.2),
                          (YOLOv7SingleDetectionRunner(Args(classes=[tracking_cls], img_size=128)), 180, 0.05),
                          (YOLOv7SingleDetectionRunner(Args(classes=[tracking_cls], img_size=64)), 120, 0.0125),
                          (YOLOv7SingleDetectionRunner(Args(classes=[tracking_cls], img_size=32)), 80, 0.0)]
        self.tracking_cls = tracking_cls

    def is_available(self, timestamp: int) -> bool:
        return timestamp - self.next_inference_timestamp >= 0

    def update_velocities_and_last_good_detection(self, detection: Box, timestamp: int):
        dt = timestamp - self.last_good_result_timestamp
        self.center_velocity = ((detection.x - self.last_good_result.x) * self.velocity_coef / dt, (detection.y - self.last_good_result.y) * self.velocity_coef / dt)
        self.prediction_area_size_velocity = (log(2 / max(detection.w, detection.h)) / self.max_delta_time, log(max(detection.w, detection.h)))
        self.last_good_result_timestamp = timestamp
        self.last_good_result = detection

    def get_detection_area_for_inference(self, timestamp: int):
        bounded_dt = min(self.max_delta_time, timestamp - self.last_good_result_timestamp)
        side_size = min(2.0, exp(min(1.0, self.prediction_area_size_velocity[0] * bounded_dt + self.prediction_area_size_velocity[1])))
        prediction_box = Box(max(0.0, min(1.0, self.last_good_result.x + (timestamp - self.last_good_result_timestamp) * self.center_velocity[0])),
                             max(0.0, min(1.0, self.last_good_result.y + (timestamp - self.last_good_result_timestamp) * self.center_velocity[1])),
                             side_size, side_size)
        left = max(0.0, min(1.0, prediction_box.x - prediction_box.w / 2))
        right = max(0.0, min(1.0, prediction_box.x + prediction_box.w / 2))
        top = max(0.0, min(1.0, prediction_box.y - prediction_box.h / 2))
        bottom = max(0.0, min(1.0, prediction_box.y + prediction_box.h / 2))
        fixed_prediction_box = Box((right + left) / 2, (top + bottom) / 2, right - left, bottom - top)
        return fixed_prediction_box

    def get_detector_index_by_detection_area(self, detection_area: Box):
        square = detection_area.w * detection_area.h
        for i in range(len(self.detectors)):
            if self.detectors[i][2] <= square:
                return i

    def process_frame(self, frame, timestamp: int) -> Box:
        detection_area = self.get_detection_area_for_inference(timestamp)
        detector_index = self.get_detector_index_by_detection_area(detection_area)

        cropped_frame = get_cropped_image(frame, detection_area)

        results = self.detectors[detector_index][0].run(cropped_frame)
        self.next_inference_timestamp = timestamp + self.detectors[detector_index][1]
        if len(results) == 0:
            return None

        best_result = max(results, key=lambda detection: detection.conf).box
        transform_to_absolute_from_relative(best_result, detection_area)
        self.update_velocities_and_last_good_detection(best_result, timestamp)

        return best_result

    def get_prediction_area(self, timestamp: int) -> Box:
        return self.get_detection_area_for_inference(timestamp)


if __name__ == "__main__":
    tracker = YOLOv7ManualTracker()
    run_solution(tracker, "inference/1.mp4", "inference/1_raw.json")
