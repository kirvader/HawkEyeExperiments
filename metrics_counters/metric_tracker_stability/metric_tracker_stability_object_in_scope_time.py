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


def compare_frame_results(considering_result: FrameProcessingInfo, state_of_art_result: FrameProcessingInfo, eps=0.15):
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
    def __init__(self, data: list, possible_max: float):
        self.data = data
        self.possible_max = possible_max

    def plus_data(self, other):
        for i in range(len(self.data)):
            self.data[i] += other.data[i]

    def multiply_data(self, multiplier: float):
        for i in range(len(self.data)):
            self.data[i] *= multiplier

    def to_dict(self):
        return {
            "data": self.data,
            "possible_max": self.possible_max
        }

    def normalize(self):
        for i in range(len(self.data)):
            self.data[i] *= 100.0 / self.possible_max
        self.possible_max = 100.0

    @staticmethod
    def from_dict(src: dict):
        return StabilityResults(src["data"], src["possible_max"])


def plot_possible_max_for_stability_results(xs_length: int, possible_max: float, fps: float):
    plt.plot(list(map(lambda item: item / fps, range(xs_length))),
             [possible_max for _ in range(xs_length)], label="Possible max", linewidth=3)


def plot_stability_results(results: StabilityResults, label: str, fps: float, linewidth=1.5):
    plt.plot(list(map(lambda item: item / fps, range(len(results.data)))), results.data, label=label, linewidth=linewidth)


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

        frames_since_last_good_detection = [0 for _ in range(self.max_lost_object_on_consecutive_frames)]
        overall_frames_quantity_when_object_lost = [0 for _ in range(self.max_lost_object_on_consecutive_frames)]

        s1 = 0
        s2 = 0
        while s1 < len(considering_data) and s2 < len(state_of_art_data):
            index_compare_result = considering_data[s1].frame_index - state_of_art_data[s2].frame_index
            if index_compare_result < 0:
                s1 += 1
                continue
            if index_compare_result > 0:
                for i in range(self.max_lost_object_on_consecutive_frames):
                    frames_since_last_good_detection[i] += 1
                s2 += 1
                continue
            current_pair_result = compare_frame_results(considering_data[s1], state_of_art_data[s2])
            if current_pair_result == DetectionType.PositiveTrue:
                for i in range(self.max_lost_object_on_consecutive_frames):
                    if frames_since_last_good_detection[i] > i:
                        overall_frames_quantity_when_object_lost[i] += frames_since_last_good_detection[i]
                    frames_since_last_good_detection[i] = 0
            else:
                for i in range(self.max_lost_object_on_consecutive_frames):
                    frames_since_last_good_detection[i] += 1
            s1 += 1
            s2 += 1
        for i in range(self.max_lost_object_on_consecutive_frames):
            if frames_since_last_good_detection[i] > i:
                overall_frames_quantity_when_object_lost[i] += frames_since_last_good_detection[i]
            frames_since_last_good_detection[i] = 0

        stability_results = StabilityResults(list(
            map(lambda item: (len(state_of_art_data) - item) / self.fps, overall_frames_quantity_when_object_lost)),
            len(state_of_art_data) / self.fps)

        with open(str(current_comparison_output_directory / "raw.json"), "w") as f:
            f.write(json.dumps(stability_results.to_dict(), indent=4))

        plot_possible_max_for_stability_results(self.max_lost_object_on_consecutive_frames,
                                                stability_results.possible_max, self.fps)
        plot_stability_results(stability_results, current_tracker_with_config, self.fps)
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
