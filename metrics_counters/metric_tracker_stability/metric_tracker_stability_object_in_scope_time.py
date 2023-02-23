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

def plot_possible_maximum(max_value, xs_length):
    plt.plot(list(map(lambda item: item / 30, range(xs_length))), [max_value for _ in range(xs_length)])

def plot_tracker_stability_graphics(points, filename):
    plt.plot(list(map(lambda item: item / 30, range(len(points)))), points)
    plt.savefig(filename)
    plt.close()

class MetricTrackerStabilityObjectInScopeTime(MetricCounterBase):
    METRIC_NAME = "METRIC_TRACKER_STABILITY_OBJECT_IN_SCOPE_TIME"

    def plot_all_on_one(self, results_folder, tracker_names, video_names, max_columns=-1):
        columns = len(video_names)
        if max_columns != -1:
            columns = min(max_columns, len(video_names))
        rows = (len(video_names) + columns - 1) // columns
        result_figure, ax = plt.subplots(rows, columns, figsize=(20, 20))

        for video_index in range(len(video_names)):
            index_in_grid = video_index // columns, video_index % columns
            ax[index_in_grid].set_title(f"{video_names[video_index]}.mp4", fontdict={"fontsize": 20})
            ax[index_in_grid].set_xlabel("Time when object is in focus", fontdict={"fontsize": 20})
            ax[index_in_grid].set_ylabel("Time object can be not detected without losing the focus", fontdict={"fontsize": 20})
            max_possible_value = -1
            data_length = -1
            for tracker_name in tracker_names:
                raw_data_file = open(Path(results_folder) / tracker_name / video_names[video_index] / f"{MetricTrackerStabilityObjectInScopeTime.METRIC_NAME}_raw.json")
                json_obj = json.load(raw_data_file)
                raw_data_file.close()
                max_possible_value = json_obj["possible_max"]
                data_length = len(json_obj["data"])
                ax[index_in_grid].plot(list(map(lambda item: item / 30, range(data_length))), json_obj["data"], label=tracker_name)
                ax[index_in_grid].legend()
            ax[index_in_grid].plot(list(map(lambda item: item / 30, range(data_length))), [max_possible_value for _ in range(data_length)], label="maximum")
            ax[index_in_grid].legend()

        trackers_string = ", ".join(tracker_names)

        filename = str(Path(results_folder) / f"{MetricTrackerStabilityObjectInScopeTime.METRIC_NAME}_all_charts_on_one.pdf")
        plt.savefig(filename)
        plt.close()
        print(f"All results for metric {MetricTrackerStabilityObjectInScopeTime.METRIC_NAME} of trackers [{trackers_string}] are saved to {filename}")

    def count(self, raw_considering_results_filename: str, raw_state_of_art_results_filename: str):
        considering_results_file = open(raw_considering_results_filename)
        considering_data = list(
            map(lambda item: FrameProcessingInfo.from_dict(item), json.load(considering_results_file)))
        considering_results_file.close()

        state_of_art_results_file = open(raw_state_of_art_results_filename)
        state_of_art_data = list(
            map(lambda item: FrameProcessingInfo.from_dict(item), json.load(state_of_art_results_file)))
        state_of_art_results_file.close()


        max_lost_object_on_consecutive_frames = 60
        frames_since_last_good_detection = [0 for i in range(max_lost_object_on_consecutive_frames)]
        overall_frames_quantity_when_object_lost = [0 for i in range(max_lost_object_on_consecutive_frames)]

        s1 = 0
        s2 = 0
        while s1 < len(considering_data) and s2 < len(state_of_art_data):
            index_compare_result = considering_data[s1].frame_index - state_of_art_data[s2].frame_index
            if index_compare_result < 0:
                s1 += 1
                continue
            if index_compare_result > 0:
                for i in range(max_lost_object_on_consecutive_frames):
                    frames_since_last_good_detection[i] += 1
                s2 += 1
                continue
            current_pair_result = compare_frame_results(considering_data[s1], state_of_art_data[s2])
            if current_pair_result == DetectionType.PositiveTrue:
                for i in range(max_lost_object_on_consecutive_frames):
                    if frames_since_last_good_detection[i] > i:
                        overall_frames_quantity_when_object_lost[i] += frames_since_last_good_detection[i]
                    frames_since_last_good_detection[i] = 0
            else:
                for i in range(max_lost_object_on_consecutive_frames):
                    frames_since_last_good_detection[i] += 1
            s1 += 1
            s2 += 1
        for i in range(max_lost_object_on_consecutive_frames):
            if frames_since_last_good_detection[i] > i:
                overall_frames_quantity_when_object_lost[i] += frames_since_last_good_detection[i]
            frames_since_last_good_detection[i] = 0

        overall_frames_quantity_when_object_lost = list(map(lambda item: (len(state_of_art_data) - item) / 30, overall_frames_quantity_when_object_lost))

        with open(str(Path(raw_considering_results_filename).parent / f"{MetricTrackerStabilityObjectInScopeTime.METRIC_NAME}_raw.json"), "w") as f:
            f.write(json.dumps({"data": overall_frames_quantity_when_object_lost, "possible_max": len(state_of_art_data) / 30}))
            print(f"Raw results for metric stability {MetricTrackerStabilityObjectInScopeTime.METRIC_NAME} of comparing {raw_considering_results_filename} to {raw_state_of_art_results_filename} is saved to {MetricTrackerStabilityObjectInScopeTime.METRIC_NAME}_raw.json")

        filename = str(Path(
            raw_considering_results_filename).parent / f"{MetricTrackerStabilityObjectInScopeTime.METRIC_NAME}.png")

        plot_possible_maximum(len(state_of_art_data) / 30, max_lost_object_on_consecutive_frames)
        plot_tracker_stability_graphics(overall_frames_quantity_when_object_lost, filename)
        print(
            f"Result for metric stability {MetricTrackerStabilityObjectInScopeTime.METRIC_NAME} of comparing {raw_considering_results_filename} to {raw_state_of_art_results_filename} is saved to {filename}")


if __name__ == "__main__":
    MetricTrackerStabilityObjectInScopeTime().count("/home/kir/hawk-eye/HawkEyeExperiments/temp/a/1/raw.json",
                                   "/home/kir/hawk-eye/HawkEyeExperiments/temp/b/1/raw.json")
