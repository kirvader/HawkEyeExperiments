import json
import sys
from enum import Enum
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

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
        if considering_result.detection_box is None:
            return DetectionType.NegativeTrue
        else:
            return DetectionType.NegativeFalse
    else:
        if considering_result.detection_box is None or not considering_result.detection_box.is_close_to(
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
        print(self.data)
        return {
            "data": self.data,
        }

    @staticmethod
    def from_dict(src: dict):
        return StabilityResults(src["data"])


def plot_stability_results(ax, results: StabilityResults, bounds: list, width=0.6, delta_x=-0.3, color="b"):
    extended_bounds = bounds + [bounds[-1] * 1.3]
    old_bound_index = -1
    current_bound_index = -1

    bar_sizes = []
    for bound in extended_bounds:
        while current_bound_index + 1 < len(results.data) and results.data[current_bound_index + 1] <= bound:
            current_bound_index += 1
        quantity_in_current_bounds = current_bound_index - old_bound_index
        bar_sizes.append(quantity_in_current_bounds)
        old_bound_index = current_bound_index
    ax.bar(np.arange(len(bounds) + 1) + delta_x, bar_sizes, width=width, color=color)


def save_results(str_bounds_to_show: list, filename: str):
    plt.xticks(range(len(str_bounds_to_show)), str_bounds_to_show)
    plt.title("Stability metric")
    plt.xlabel("Bounds, ms")
    plt.ylabel("Quantity, times")

    plt.savefig(filename)
    plt.close()


class MetricTrackerStabilityObjectInScopeTime(MetricCounterBase):
    METRIC_NAME = "METRIC_TRACKER_STABILITY_OBJECT_IN_SCOPE_TIME"

    def __init__(self, bounds=None, fps=30):
        if bounds is None:
            bounds = [60, 120, 200, 300, 400, 500, 600, 800, 1000, 1200]
        self.fps = fps
        self.bounds = bounds
        self.general_colors = [
            'b', 'g', 'r', 'c', 'm', 'y', 'k'
        ]

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
        width = 0.7 / len(tracker_names_with_configs)
        colors = {}

        fig = plt.figure()
        ax = plt.subplot(111)

        for i in range(len(tracker_names_with_configs)):
            tracker_name_with_config = tracker_names_with_configs[i]
            colors[tracker_name_with_config] = self.general_colors[i % len(self.general_colors)]
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
            plot_stability_results(ax, current_tracker_with_config_stability_results, self.bounds, width,
                                   -(width / 2 + width * i), colors[tracker_name_with_config])

        labels = list(colors.keys())
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height * 0.7])

        plt.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.0, 1))

        bounds_to_show = list(map(str, self.bounds))
        bounds_to_show.append(f">{self.bounds[-1]}")

        save_results(bounds_to_show, str(current_comparison_output_directory / "stability.png"))

    def count_average_across_many_videos(self,
                                         metrics_raw_results_folder: str,
                                         tracker_name_with_config: str,
                                         state_of_art_tracker_with_config: str,
                                         video_names: list):
        pass

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

        time_deltas_between_good_estimations = []  # good estimation = estimation which is close to real result.
        last_good_detection_frame_index = -1
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
                time_since_last_good_estimation = (considering_data[
                                                       s1].frame_index - last_good_detection_frame_index) * 1000 / self.fps
                time_deltas_between_good_estimations.append(time_since_last_good_estimation)
                last_good_detection_frame_index = considering_data[s1].frame_index
            s1 += 1
            s2 += 1

        fig = plt.figure()
        ax = plt.subplot(111)

        time_deltas_between_good_estimations.sort()
        stability_results = StabilityResults(time_deltas_between_good_estimations)

        with open(str(current_comparison_output_directory / "raw.json"), "w") as f:
            f.write(json.dumps(stability_results.to_dict(), indent=4))

        plot_stability_results(ax, stability_results, self.bounds)

        bounds_to_show = list(map(str, self.bounds))
        bounds_to_show.append(f">{self.bounds[-1]}")

        save_results(bounds_to_show, str(current_comparison_output_directory / f"stability.png"))


if __name__ == "__main__":
    MetricTrackerStabilityObjectInScopeTime().count_tracker_performance_on_single_video(
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/raw_results/",
        "marked/manually",
        "manual_tracking_with_yolov7/with_speed_dec_0_5_est_coef_0_7",
        "1_slow_ball_with_shadow",
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/metrics_results")
    MetricTrackerStabilityObjectInScopeTime().count_tracker_performance_on_single_video(
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/raw_results/",
        "marked/manually",
        "manual_tracking_with_yolov7/with_speed_dec_0_7_est_coef_0_7",
        "1_slow_ball_with_shadow",
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/metrics_results")
    MetricTrackerStabilityObjectInScopeTime().count_tracker_performance_on_single_video(
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/raw_results/",
        "marked/manually",
        "manual_tracking_with_yolov7/with_speed_dec_1_0_est_coef_0_7",
        "1_slow_ball_with_shadow",
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/metrics_results")
    MetricTrackerStabilityObjectInScopeTime().count_tracker_performance_on_single_video(
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/raw_results/",
        "marked/manually",
        "manual_tracking_with_yolov7/no_speed",
        "1_slow_ball_with_shadow",
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/metrics_results")
    MetricTrackerStabilityObjectInScopeTime().count_tracker_performance_on_single_video(
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/raw_results/",
        "marked/manually",
        "pure_yolov7_detector/default_with_natural_scale",
        "1_slow_ball_with_shadow",
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/metrics_results")
    MetricTrackerStabilityObjectInScopeTime().count_tracker_performance_on_single_video(
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/raw_results/",
        "marked/manually",
        "pure_yolov7_detector/state_of_art_with_natural_scale",
        "1_slow_ball_with_shadow",
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/metrics_results")
    MetricTrackerStabilityObjectInScopeTime().compare_trackers_on_single_video(
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/metrics_results",
        [
            "manual_tracking_with_yolov7/no_speed",
            "manual_tracking_with_yolov7/with_speed_dec_0_5_est_coef_0_7",
            "manual_tracking_with_yolov7/with_speed_dec_0_7_est_coef_0_7",
            "manual_tracking_with_yolov7/with_speed_dec_1_0_est_coef_0_7",
            "pure_yolov7_detector/default_with_natural_scale",
            "pure_yolov7_detector/state_of_art_with_natural_scale",
        ],
        "marked/manually",
        "1_slow_ball_with_shadow",
    )

    # MetricTrackerStabilityObjectInScopeTime().count_tracker_performance_on_single_video(
    #     "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/raw_results/",
    #     "pure_yolov7_detector/state_of_art",
    #     "pure_yolov7_detector/state_of_art",
    #     "1",
    #     "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/metrics_results")
    # MetricTrackerStabilityObjectInScopeTime().count_tracker_performance_on_single_video(
    #     "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/raw_results/",
    #     "pure_yolov7_detector/state_of_art",
    #     "pure_yolov7_detector/default",
    #     "1",
    #     "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/metrics_results")
    # MetricTrackerStabilityObjectInScopeTime().count_tracker_performance_on_single_video(
    #     "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/raw_results/",
    #     "pure_yolov7_detector/state_of_art",
    #     "pure_yolov7_detector/default",
    #     "2",
    #     "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/metrics_results")
    # MetricTrackerStabilityObjectInScopeTime().count_tracker_performance_on_single_video(
    #     "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/raw_results/",
    #     "pure_yolov7_detector/state_of_art",
    #     "pure_yolov7_detector/default",
    #     "3",
    #     "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/metrics_results")
    # MetricTrackerStabilityObjectInScopeTime().count_tracker_performance_on_single_video(
    #     "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/raw_results/",
    #     "pure_yolov7_detector/state_of_art",
    #     "manual_tracking_with_yolov7/no_speed",
    #     "1",
    #     "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/metrics_results")
    # MetricTrackerStabilityObjectInScopeTime().count_tracker_performance_on_single_video(
    #     "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/raw_results/",
    #     "pure_yolov7_detector/state_of_art",
    #     "manual_tracking_with_yolov7/with_speed",
    #     "1",
    #     "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/metrics_results")
    # MetricTrackerStabilityObjectInScopeTime().compare_trackers_on_single_video(
    #     "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/metrics_results",
    #     [
    #         "pure_yolov7_detector/state_of_art",
    #         "pure_yolov7_detector/default",
    #         "manual_tracking_with_yolov7/no_speed",
    #         "manual_tracking_with_yolov7/with_speed"
    #     ],
    #     "pure_yolov7_detector/state_of_art",
    #     "1"
    # )
    # MetricTrackerStabilityObjectInScopeTime().count_average_across_many_videos(
    #     "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/metrics_results",
    #     "pure_yolov7_detector/default",
    #     "pure_yolov7_detector/state_of_art",
    #     ["1", "2", "3"]
    # )
