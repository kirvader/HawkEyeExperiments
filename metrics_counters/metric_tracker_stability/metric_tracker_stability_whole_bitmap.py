import json
import sys
from enum import Enum
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import image as mpimg

from utils.detection_result import Box

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from metrics_counters.metric_counter_base import MetricCounterBase
from utils.frame_processing_info import FrameProcessingInfo


class DetectionType(Enum):
    PositiveTrue = 1
    NegativeTrue = 0
    PositiveFalse = -1
    NegativeFalse = -2

    @staticmethod
    def from_value(value):
        if value == 1:
            return DetectionType.PositiveTrue
        elif value == 0:
            return DetectionType.NegativeTrue
        elif value == -1:
            return DetectionType.PositiveFalse
        elif value == -2:
            return DetectionType.NegativeFalse


def compare_frame_results(considering_result: FrameProcessingInfo, state_of_art_result: FrameProcessingInfo, eps=0.05):
    if state_of_art_result.detection_result is None:
        if considering_result.detection_result is None:
            return DetectionType.NegativeTrue
        else:
            return DetectionType.NegativeFalse
    else:
        if considering_result.detection_result is None or not considering_result.detection_result.is_close_to(
                state_of_art_result.detection_result, eps):
            return DetectionType.PositiveFalse
        else:
            return DetectionType.PositiveTrue


def plot_single_horizontal_chart(videos, fst_part, snd_part):
    frames_handled = [fst_part]
    all_frames_count = [snd_part]

    b1 = plt.barh(videos, frames_handled, color="blue")

    b2 = plt.barh(videos, all_frames_count, left=frames_handled, color="gray")

    plt.legend([b1, b2], ["Handled", "Not handled"], title="Efficiency", loc="upper right")


class Results:
    def __init__(self, sequence_detections=None, comparing_file=""):
        if sequence_detections is None:
            sequence_detections = []
        self.sequence_detections = sequence_detections
        self.comparing_file = comparing_file

    def add_frame_result(self, unit_result: DetectionType):
        self.sequence_detections.append(unit_result)

    def to_dict(self):
        return {
            "sequence_detections": list(map(lambda item: item.value, self.sequence_detections)),
            "comparing_file": self.comparing_file
        }

    def plot(self, filename):
        x = range(len(self.sequence_detections))
        y = [item.value for item in self.sequence_detections]
        def color_by_detection_type(detection_type: DetectionType):
            if detection_type == DetectionType.PositiveTrue or detection_type == DetectionType.NegativeTrue:
                return "green"
            elif detection_type == DetectionType.PositiveFalse or detection_type == DetectionType.NegativeFalse:
                return "red"

        colors = [color_by_detection_type(item) for item in self.sequence_detections]
        plt.scatter(x, y, c=colors, s=0.01)
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def from_dict(data):
        return Results(list(map(lambda item: DetectionType.from_value(item), data["sequence_detections"])),
                       data["comparing_file"])


class MetricTrackerStabilityWholeBitmap(MetricCounterBase):
    METRIC_NAME = "METRIC_TRACKER_STABILITY_WHOLE_BITMAP"

    def plot_all_on_one(self, results_folder, tracker_names, video_names, max_columns=-1):
        columns = len(video_names)
        if max_columns != -1:
            columns = min(max_columns, len(video_names))
        rows = (len(video_names) + columns - 1) // columns
        rows *= len(tracker_names)
        result_figure = plt.figure(figsize=(40, 40))

        for tracker_index in range(len(tracker_names)):
            tracker_folder_path = Path(results_folder) / tracker_names[tracker_index]
            for video_index in range(len(video_names)):
                index_in_grid = tracker_index * columns + video_index + 1
                filename = tracker_folder_path / video_names[
                    video_index] / f"{MetricTrackerStabilityWholeBitmap.METRIC_NAME}_true_false_positive_negative.png"
                img = mpimg.imread(str(filename))
                result_figure.add_subplot(rows, columns, index_in_grid)
                plt.imshow(img)

        trackers_string = ", ".join(tracker_names)

        filename = str(Path(results_folder) / f"{MetricTrackerStabilityWholeBitmap.METRIC_NAME}_all_charts_on_one.pdf")
        plt.savefig(filename)
        plt.close()
        print(f"All results for metric {MetricTrackerStabilityWholeBitmap.METRIC_NAME} of trackers [{trackers_string}] are saved to {filename}")

    def count(self, raw_considering_results_filename: str, raw_state_of_art_results_filename: str):
        considering_results_file = open(raw_considering_results_filename)
        considering_data = list(
            map(lambda item: FrameProcessingInfo.from_dict(item), json.load(considering_results_file)))
        considering_results_file.close()

        state_of_art_results_file = open(raw_state_of_art_results_filename)
        state_of_art_data = list(
            map(lambda item: FrameProcessingInfo.from_dict(item), json.load(state_of_art_results_file)))
        state_of_art_results_file.close()

        results = Results([], raw_state_of_art_results_filename)

        s1 = 0
        s2 = 0
        while s1 < len(considering_data) and s2 < len(state_of_art_data):
            index_compare_result = considering_data[s1].frame_index - state_of_art_data[s2].frame_index
            if index_compare_result < 0:
                s1 += 1
                continue
            if index_compare_result > 0:
                result = compare_frame_results(FrameProcessingInfo(state_of_art_data[s2].frame_index, Box(0.5, 0.5, 1.0, 1.0), None),
                                                               state_of_art_data[s2])
                results.add_frame_result(result)
                s2 += 1
                continue
            current_pair_result = compare_frame_results(considering_data[s1], state_of_art_data[s2])
            results.add_frame_result(current_pair_result)
            s1 += 1
            s2 += 1

        with open(Path(raw_considering_results_filename).parent / f"{MetricTrackerStabilityWholeBitmap.METRIC_NAME}.json", "w") as json_file:
            json_file.write(json.dumps(results.to_dict(), indent=4))

        filename = str(Path(raw_considering_results_filename).parent / f"{MetricTrackerStabilityWholeBitmap.METRIC_NAME}_true_false_positive_negative.png")
        results.plot(filename)
        print(f"Result for metric true_false_positive_negative {MetricTrackerStabilityWholeBitmap.METRIC_NAME} of comparing {raw_considering_results_filename} to {raw_state_of_art_results_filename} is saved to {filename}")


if __name__ == "__main__":
    MetricTrackerStabilityWholeBitmap().count("/home/kir/hawk-eye/HawkEyeExperiments/temp/a/1/raw.json",
                                   "/home/kir/hawk-eye/HawkEyeExperiments/temp/b/1/raw.json")
