import json
import sys
from enum import Enum
from pathlib import Path
import matplotlib.pyplot as plt

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
    if state_of_art_result.detection_box is None:
        if considering_result.real_time_estimate_box is None:
            return DetectionType.NegativeTrue
        else:
            return DetectionType.NegativeFalse
    else:
        if considering_result.real_time_estimate_box is None or not considering_result.real_time_estimate_box.is_close_to(
                state_of_art_result.detection_box, eps):
            return DetectionType.PositiveFalse
        else:
            return DetectionType.PositiveTrue


class StabilityResults:
    def __init__(self, data: list):
        self.data = data

    def plus_data(self, other):
        for i in range(len(self.data)):
            self.data[i] += other.data[i]

    def multiply_data(self, multiplier: float):
        for i in range(len(self.data)):
            self.data[i] *= multiplier

    def to_dict(self):
        return {
            "data": self.data,
        }

    @staticmethod
    def from_dict(src: dict):
        return StabilityResults(src["data"])



def plot_stability_results(results: StabilityResults, bounds: list):
    extended_bounds = bounds + [1000000]
    old_bound_index = -1
    current_bound_index = -1

    bar_sizes = []
    for bound in extended_bounds:
        while current_bound_index + 1 < len(results.data) and results.data[current_bound_index + 1] <= bound:
            current_bound_index += 1
        quantity_in_current_bounds = current_bound_index - old_bound_index
        bar_sizes.append(quantity_in_current_bounds)
        old_bound_index = current_bound_index
    bounds_to_show = list(map(str, bounds))
    bounds_to_show.append(f">{bounds[-1]}")
    plt.bar(bounds_to_show, bar_sizes)


def save_plot_for_stability_resutls(is_normalized: bool, filename: str):
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    if is_normalized:
        plt.title("Stability(normalized)")
        plt.xlabel("Focus bound, s")
        plt.ylabel("Time approved, %")
    else:
        plt.title("Stability")
        plt.xlabel("Focus bound, s")
        plt.ylabel("Time approved, s")
    plt.savefig(filename)
    plt.close()


class MetricTrackerStabilityObjectInScopeTime(MetricCounterBase):
    METRIC_NAME = "METRIC_TRACKER_STABILITY_OBJECT_IN_SCOPE_TIME"

    def __init__(self, max_lost_object_on_consecutive_frames=60, fps=30):
        self.max_lost_object_on_consecutive_frames = max_lost_object_on_consecutive_frames
        self.fps = fps

    def compare_trackers_on_single_video(self,
                                         metrics_output_directory: str,
                                         tracker_names_with_configs: list,
                                         state_of_art_tracker_with_config: str,
                                         video_name: str):
        current_comparison_output_directory = MetricCounterBase.get_output_folder_for_many_trackers_single_video(
            state_of_art_tracker_with_config,
            tracker_names_with_configs,
            video_name,
            metrics_output_directory,
            MetricTrackerStabilityObjectInScopeTime.METRIC_NAME
        )

        possible_max = -1
        for tracker_name_with_config in tracker_names_with_configs:
            current_tracker_metric_raw_results_file = MetricCounterBase.get_output_folder_for_single_tracker_single_video(
                state_of_art_tracker_with_config,
                tracker_name_with_config,
                video_name,
                metrics_output_directory,
                MetricTrackerStabilityObjectInScopeTime.METRIC_NAME
            ) / "raw.json"
            with open(current_tracker_metric_raw_results_file) as f:
                json_object = json.load(f)
            current_tracker_with_config_stability_results = StabilityResults.from_dict(json_object)
            possible_max = current_tracker_with_config_stability_results.possible_max
            plot_stability_results(current_tracker_with_config_stability_results, tracker_name_with_config, self.fps)
        plot_possible_max_for_stability_results(self.max_lost_object_on_consecutive_frames, possible_max, self.fps)
        save_plot_for_stability_resutls(False, str(current_comparison_output_directory / "stability.png"))

    def count_average_across_many_videos(self,
                                         metrics_raw_results_folder: str,
                                         tracker_name_with_config: str,
                                         state_of_art_tracker_with_config: str,
                                         video_names: list):
        current_output_folder = MetricCounterBase.get_output_folder_for_single_tracker_many_video(
            state_of_art_tracker_with_config, tracker_name_with_config, video_names, metrics_raw_results_folder,
            MetricTrackerStabilityObjectInScopeTime.METRIC_NAME)

        average = StabilityResults([0 for _ in range(self.max_lost_object_on_consecutive_frames)], 100.0)
        plot_possible_max_for_stability_results(self.max_lost_object_on_consecutive_frames, 100.0, self.fps)
        for video_name in video_names:
            current_tracker_metric_raw_results_file = MetricCounterBase.get_output_folder_for_single_tracker_single_video(
                state_of_art_tracker_with_config,
                tracker_name_with_config,
                video_name,
                metrics_raw_results_folder,
                MetricTrackerStabilityObjectInScopeTime.METRIC_NAME
            ) / "raw.json"
            with open(current_tracker_metric_raw_results_file) as f:
                json_object = json.load(f)
            current_tracker_with_config_stability_results = StabilityResults.from_dict(json_object)
            current_tracker_with_config_stability_results.normalize()
            average.plus_data(current_tracker_with_config_stability_results)
            plot_stability_results(current_tracker_with_config_stability_results, video_name, self.fps)
        average.multiply_data(1 / len(video_names))
        plot_stability_results(average, "Average", self.fps, 4.5)
        save_plot_for_stability_resutls(True, str(current_output_folder / "stability.png"))

    def count_tracker_performance_on_single_video(self,
                                                  raw_results_directory: str,
                                                  state_of_art_tracker_with_config: str,
                                                  current_tracker_with_config: str,
                                                  video_name: str,
                                                  metrics_output_directory: str):
        current_comparison_output_directory = MetricCounterBase.get_output_folder_for_single_tracker_single_video(
            state_of_art_tracker_with_config,
            current_tracker_with_config,
            video_name,
            metrics_output_directory,
            MetricTrackerStabilityObjectInScopeTime.METRIC_NAME
        )

        considering_data = MetricCounterBase.read_raw_results(
            raw_results_directory,
            current_tracker_with_config,
            video_name
        )

        state_of_art_data = MetricCounterBase.read_raw_results(
            raw_results_directory,
            state_of_art_tracker_with_config,
            video_name
        )

        time_deltas_between_good_estimations = [] #  good estimation = estimation which is close to real result.
        last_good_detection_frame_index = 0
        s1 = 0
        s2 = 0
        while s1 < len(considering_data) and s2 < len(state_of_art_data):
            index_compare_result = considering_data[s1].frame_index - state_of_art_data[s2].frame_index
            if index_compare_result < 0:
                s1 += 1
                continue
            if index_compare_result > 0:
                s2 += 1
                continue
            current_frame_estimation_rating = compare_frame_results(considering_data[s1], state_of_art_data[s2])
            if current_frame_estimation_rating == DetectionType.PositiveTrue:
                time_since_last_good_estimation = (considering_data[s1].frame_index - last_good_detection_frame_index) / self.fps
                time_deltas_between_good_estimations.append(time_since_last_good_estimation)
                last_good_detection_frame_index = considering_data[s1].frame_index

        time_deltas_between_good_estimations.sort()
        stability_results = StabilityResults(time_deltas_between_good_estimations)

        with open(str(current_comparison_output_directory / "raw.json"), "w") as f:
            f.write(json.dumps(stability_results.to_dict(), indent=4))

        plot_stability_results(stability_results, current_tracker_with_config)
        save_plot_for_stability_resutls(False, str(current_comparison_output_directory / f"stability.png"))


if __name__ == "__main__":
    MetricTrackerStabilityObjectInScopeTime().count_tracker_performance_on_single_video(
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/raw_results/",
        "pure_yolov7_detector/state_of_art",
        "pure_yolov7_detector/state_of_art",
        "1",
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/metrics_results")

    MetricTrackerStabilityObjectInScopeTime().count_tracker_performance_on_single_video(
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/raw_results/",
        "pure_yolov7_detector/state_of_art",
        "pure_yolov7_detector/state_of_art",
        "1",
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/metrics_results")
    MetricTrackerStabilityObjectInScopeTime().count_tracker_performance_on_single_video(
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/raw_results/",
        "pure_yolov7_detector/state_of_art",
        "pure_yolov7_detector/default",
        "1",
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/metrics_results")
    MetricTrackerStabilityObjectInScopeTime().count_tracker_performance_on_single_video(
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/raw_results/",
        "pure_yolov7_detector/state_of_art",
        "pure_yolov7_detector/default",
        "2",
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/metrics_results")
    MetricTrackerStabilityObjectInScopeTime().count_tracker_performance_on_single_video(
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/raw_results/",
        "pure_yolov7_detector/state_of_art",
        "pure_yolov7_detector/default",
        "3",
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/metrics_results")
    MetricTrackerStabilityObjectInScopeTime().count_tracker_performance_on_single_video(
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/raw_results/",
        "pure_yolov7_detector/state_of_art",
        "manual_tracking_with_yolov7/no_speed",
        "1",
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/metrics_results")
    MetricTrackerStabilityObjectInScopeTime().count_tracker_performance_on_single_video(
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/raw_results/",
        "pure_yolov7_detector/state_of_art",
        "manual_tracking_with_yolov7/with_speed",
        "1",
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/metrics_results")
    MetricTrackerStabilityObjectInScopeTime().compare_trackers_on_single_video(
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/metrics_results",
        [
            "pure_yolov7_detector/state_of_art",
            "pure_yolov7_detector/default",
            "manual_tracking_with_yolov7/no_speed",
            "manual_tracking_with_yolov7/with_speed"
        ],
        "pure_yolov7_detector/state_of_art",
        "1"
    )
    MetricTrackerStabilityObjectInScopeTime().count_average_across_many_videos(
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/metrics_results",
        "pure_yolov7_detector/default",
        "pure_yolov7_detector/state_of_art",
        ["1", "2", "3"]
    )
