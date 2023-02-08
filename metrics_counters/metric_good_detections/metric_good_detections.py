import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from metrics_counters.metric_counter_base import MetricCounterBase
from utils.frame_processing_info import FrameProcessingInfo


class DetectionCounter:
    def __init__(self):
        self.detected_right = 0
        self.detected_wrong = 0
        self.not_detected_right = 0
        self.not_detected_wrong = 0

    def plot(self, filename_output):
        values = [self.detected_right, self.not_detected_right, self.detected_wrong, self.not_detected_wrong]
        names = ["Detected right", "Not detected right", "Detected wrong", "Not detected wrong"]
        colors = ['#00ff00', '#257a25', '#ff0000', '#cc4e4e']
        plt.pie(values, labels=names, colors=colors, labeldistance=1.2)

        my_circle = plt.Circle((0, 0), 0.6, color='white')
        p = plt.gcf()
        p.gca().add_artist(my_circle)

        # plt.show()
        plt.savefig(filename_output)
        plt.close()


def compare_frame_results(considering_result: FrameProcessingInfo, state_of_art_result: FrameProcessingInfo,
                          detection_counter: DetectionCounter, eps=0.05):
    if state_of_art_result.frame_index != considering_result.frame_index:
        return considering_result.frame_index - state_of_art_result.frame_index
    if state_of_art_result.detection_result is None:
        if considering_result.detection_result is None:
            detection_counter.not_detected_right += 1
        else:
            detection_counter.not_detected_wrong += 1
    else:
        if considering_result.detection_result is None or not considering_result.detection_result.is_close_to(
                state_of_art_result.detection_result, eps):
            detection_counter.detected_wrong += 1
        else:
            detection_counter.detected_right += 1
    return 0


class MetricGoodDetections(MetricCounterBase):
    def count(self, raw_considering_results_filename: str, raw_state_of_art_results_filename: str):
        considering_results_file = open(raw_considering_results_filename)
        considering_data = list(
            map(lambda item: FrameProcessingInfo.from_dict(item), json.load(considering_results_file)))
        considering_results_file.close()

        state_of_art_results_file = open(raw_state_of_art_results_filename)
        state_of_art_data = list(
            map(lambda item: FrameProcessingInfo.from_dict(item), json.load(state_of_art_results_file)))
        state_of_art_results_file.close()

        frames_quantity = len(state_of_art_data)
        detections_counter = DetectionCounter()

        s1 = 0
        s2 = 0
        while s1 < len(considering_data) and s2 < len(state_of_art_data):
            index_compare_result = considering_data[s1].frame_index - state_of_art_data[s2].frame_index
            if index_compare_result < 0:
                s1 += 1
            elif index_compare_result > 0:
                s2 += 1
            else:
                compare_frame_results(considering_data[s1], state_of_art_data[s2], detections_counter)
                s1 += 1
                s2 += 1


        METRIC_NAME_DETECTIONS_PERCENTAGE = "METRIC_DETECTIONS_PERCENTAGE"

        with open(Path(raw_considering_results_filename).parent / f"{METRIC_NAME_DETECTIONS_PERCENTAGE}.json", "w") as json_file:
            json_file.write(json.dumps({
                "frames_quantity": frames_quantity,
                "detections_run": len(considering_data),
                "detected_true": detections_counter.detected_right,
                "detected_false": detections_counter.detected_wrong,
                "not_detected_true": detections_counter.not_detected_right,
                "not_detected_false": detections_counter.not_detected_wrong,
                "comparing_to": raw_state_of_art_results_filename
            }, indent=4))

        videos = [""]
        frames_handled = [len(considering_data)]
        all_frames_count = [frames_quantity - len(considering_data)]

        b1 = plt.barh(videos, frames_handled, color="blue")

        b2 = plt.barh(videos, all_frames_count, left=frames_handled, color="gray")

        plt.legend([b1, b2], ["Handled", "Not handled"], title="Efficiency", loc="upper right")
        plt.savefig(str(Path(raw_considering_results_filename).parent / f"{METRIC_NAME_DETECTIONS_PERCENTAGE}_efficiency.png"))
        plt.close()
        detections_counter.plot(str(Path(raw_considering_results_filename).parent / f"{METRIC_NAME_DETECTIONS_PERCENTAGE}_true_false_positive_negative.png"))



if __name__ == "__main__":
    MetricGoodDetections().count("/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/1_raw.json",
                                 "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/1_raw.json")
