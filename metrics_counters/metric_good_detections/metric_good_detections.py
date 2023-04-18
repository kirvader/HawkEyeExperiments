import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import image as mpimg

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from metrics_counters.metric_counter_base import MetricCounterBase
from utils.frame_processing_info import FrameProcessingInfo


class DetectionCounter:
    def __init__(self, detected_right=0, detected_wrong=0, not_detected_right=0, not_detected_wrong=0):
        self.detected_right = detected_right
        self.detected_wrong = detected_wrong
        self.not_detected_right = not_detected_right
        self.not_detected_wrong = not_detected_wrong

    def plot(self):
        values = [self.detected_right, self.not_detected_right, self.detected_wrong, self.not_detected_wrong]
        names = ["Detected right", "Not detected right", "Detected wrong", "Not detected wrong"]
        colors = ['#00ff00', '#257a25', '#ff0000', '#cc4e4e']
        plt.pie(values, labels=names, colors=colors, labeldistance=1.2)

        my_circle = plt.Circle((0, 0), 0.6, color='white')
        p = plt.gcf()
        p.gca().add_artist(my_circle)


def compare_frame_results(considering_result: FrameProcessingInfo, state_of_art_result: FrameProcessingInfo,
                          detection_counter: DetectionCounter, eps=0.15):
    if state_of_art_result.frame_index != considering_result.frame_index:
        return considering_result.frame_index - state_of_art_result.frame_index
    if state_of_art_result.detection_box is None:
        if considering_result.detection_box is None:
            detection_counter.not_detected_right += 1
        else:
            detection_counter.not_detected_wrong += 1
    else:
        if considering_result.detection_box is None or not considering_result.detection_box.is_close_to(
                state_of_art_result.detection_box, eps):
            detection_counter.detected_wrong += 1
        else:
            detection_counter.detected_right += 1
    return 0


def plot_single_horizontal_chart(videos, fst_part, snd_part):
    frames_handled = [fst_part]
    all_frames_count = [snd_part]

    b1 = plt.barh(videos, frames_handled, color="blue")

    b2 = plt.barh(videos, all_frames_count, left=frames_handled, color="gray")

    plt.legend([b1, b2], ["Handled", "Not handled"], title="Efficiency", loc="upper right")


class DetectionResults:
    def __init__(self, frames_quantity=0, detections_run=0, positive_true=0, positive_false=0, negative_true=0,
                 negative_false=0, comparing_file="", good_estimations_amount=0):
        self.frames_quantity = frames_quantity
        self.detections_run = detections_run
        self.positive_true = positive_true
        self.positive_false = positive_false
        self.negative_true = negative_true
        self.negative_false = negative_false
        self.comparing_file = comparing_file
        self.good_estimations_amount = good_estimations_amount

    def take_into_account(self, other):
        self.frames_quantity += other.frames_quantity
        self.detections_run += other.detections_run
        self.positive_true += other.positive_true
        self.positive_false += other.positive_false
        self.negative_true += other.negative_true
        self.negative_false += other.negative_false
        self.good_estimations_amount += other.good_estimations_amount

    def to_dict(self):
        return {
            "frames_quantity": self.frames_quantity,
            "detections_run": self.detections_run,
            "positive_true": self.positive_true,
            "positive_false": self.positive_false,
            "negative_true": self.negative_true,
            "negative_false": self.negative_false,
            "comparing_file": self.comparing_file,
            "good_estimations_amount": self.good_estimations_amount
        }

    def plot(self, filename):
        figure = plt.figure(figsize=(10, 10))
        figure.add_subplot(3, 1, 2)
        plot_single_horizontal_chart([""], self.detections_run, self.frames_quantity - self.detections_run)

        figure.add_subplot(3, 1, 3)
        plot_single_horizontal_chart([""], self.good_estimations_amount,
                                     self.frames_quantity - self.good_estimations_amount)

        figure.add_subplot(3, 1, 1)
        DetectionCounter(self.positive_true, self.positive_false, self.negative_true, self.negative_false).plot()

        plt.savefig(filename)
        plt.close()

    @staticmethod
    def from_dict(data):
        return DetectionResults(data["frames_quantity"], data["detections_run"], data["positive_true"],
                                data["positive_false"], data["negative_true"], data["negative_false"],
                                data["comparing_file"], data["good_estimations_amount"])


class MetricGoodDetections(MetricCounterBase):
    METRIC_NAME = "METRIC_DETECTIONS_PERCENTAGE"
    GENERAL_COLORS = [
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
            MetricGoodDetections.METRIC_NAME
        )

        results = {}
        for tracker_name_with_config in tracker_names_with_configs:
            current_tracker_metric_raw_results_file = MetricCounterBase.get_output_folder_for_single_tracker_single_video(
                state_of_art_tracker_with_config,
                tracker_name_with_config,
                video_name,
                metrics_output_directory,
                MetricGoodDetections.METRIC_NAME
            ) / "raw.json"
            with open(current_tracker_metric_raw_results_file) as f:
                json_object = json.load(f)
            results[tracker_name_with_config] = DetectionResults.from_dict(json_object)

        def get_detections_accuracy_list(detection_results: DetectionResults) -> list:
            return [detection_results.positive_true, detection_results.positive_false, detection_results.negative_true,
                    detection_results.negative_false]

        fig = plt.figure(figsize=(6, 5), dpi=200)
        left, bottom, width, height = 0.1, 0.3, 0.8, 0.6
        ax = fig.add_axes([left, bottom, width, height])

        width = 1 / (len(results) + 1)
        ticks = np.arange(4)
        index = -(len(results) / 2 - 1)

        int_index = 0
        colors = {}

        for tracker_name_with_config in results:
            colors[tracker_name_with_config] = MetricGoodDetections.GENERAL_COLORS[int_index % len(MetricGoodDetections.GENERAL_COLORS)]
            ax.bar(ticks + index * width, get_detections_accuracy_list(results[tracker_name_with_config]), width,
                   label=tracker_name_with_config, color=colors[tracker_name_with_config])
            index += 1
            int_index += 1

        ax.set_ylabel('Quantity of detections')
        ax.set_title('Detections accuracy')
        ax.set_xticks(ticks + width / 2)
        ax.set_xticklabels(["Positive true", "Positive false", "Negative true", "Negative false"])

        labels = list(results.keys())
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height * 0.7])

        plt.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.0, 1))
        plt.savefig(str(current_comparison_output_directory / "accuracy.png"))

    def count_average_across_many_videos(self,
                                         metrics_raw_results_folder: str,
                                         tracker_name_with_config: str,
                                         state_of_art_tracker_with_config: str,
                                         video_names: list):
        current_output_folder = MetricCounterBase.get_output_folder_for_single_tracker_many_video(
            state_of_art_tracker_with_config, tracker_name_with_config, video_names, metrics_raw_results_folder,
            MetricGoodDetections.METRIC_NAME)

        average_results = DetectionResults()
        for video_name in video_names:
            current_tracker_metric_raw_results_file = MetricCounterBase.get_output_folder_for_single_tracker_single_video(
                state_of_art_tracker_with_config,
                tracker_name_with_config,
                video_name,
                metrics_raw_results_folder,
                MetricGoodDetections.METRIC_NAME
            ) / "raw.json"
            with open(current_tracker_metric_raw_results_file) as f:
                average_results.take_into_account(DetectionResults.from_dict(json.load(f)))
        video_names_string = "-".join(video_names)
        average_results.plot(str(current_output_folder / f"average-accuracy-{video_names_string}.png"))

    def count_tracker_performance_on_single_video(self,
                                                  raw_results_directory: str,
                                                  state_of_art_tracker_with_config: str,
                                                  current_tracker_with_config: str,
                                                  video_name: str,
                                                  metrics_output_directory: str):
        good_tracker_estimations_amount = 0

        current_comparison_output_directory = MetricCounterBase.get_output_folder_for_single_tracker_single_video(
            state_of_art_tracker_with_config,
            current_tracker_with_config,
            video_name,
            metrics_output_directory,
            MetricGoodDetections.METRIC_NAME
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
                if state_of_art_data[s2].detection_box is not None and considering_data[s1].real_time_estimate_box is not None:
                    if considering_data[s1].real_time_estimate_box.is_close_to(state_of_art_data[s2].detection_box, 0.075):
                        good_tracker_estimations_amount += 1
                s1 += 1
                s2 += 1

        with open(current_comparison_output_directory / "raw.json", "w") as json_file:
            json_file.write(json.dumps(
                DetectionResults(frames_quantity, len(considering_data), detections_counter.detected_right,
                                 detections_counter.detected_wrong, detections_counter.not_detected_right,
                                 detections_counter.not_detected_wrong, state_of_art_tracker_with_config,
                                 good_tracker_estimations_amount).to_dict(),
                indent=4))

        plot_single_horizontal_chart([""], len(considering_data), frames_quantity - len(considering_data))
        filename = str(current_comparison_output_directory / "efficiency.png")
        plt.savefig(filename)
        plt.close()

        detections_counter.plot()
        filename = str(current_comparison_output_directory / "accuracy.png")
        plt.savefig(filename)
        plt.close()


if __name__ == "__main__":
    MetricGoodDetections().count_tracker_performance_on_single_video(
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/raw_results/",
        "pure_yolov7_detector/state_of_art",
        "pure_yolov7_detector/state_of_art",
        "1",
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/metrics_results")
    MetricGoodDetections().count_tracker_performance_on_single_video(
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/raw_results/",
        "pure_yolov7_detector/state_of_art",
        "pure_yolov7_detector/default",
        "1",
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/metrics_results")
    MetricGoodDetections().count_tracker_performance_on_single_video(
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/raw_results/",
        "pure_yolov7_detector/state_of_art",
        "pure_yolov7_detector/default",
        "2",
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/metrics_results")
    MetricGoodDetections().count_tracker_performance_on_single_video(
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/raw_results/",
        "pure_yolov7_detector/state_of_art",
        "pure_yolov7_detector/default",
        "3",
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/metrics_results")
    MetricGoodDetections().count_tracker_performance_on_single_video(
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/raw_results/",
        "pure_yolov7_detector/state_of_art",
        "manual_tracking_with_yolov7/no_speed",
        "1",
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/metrics_results")
    MetricGoodDetections().count_tracker_performance_on_single_video(
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/raw_results/",
        "pure_yolov7_detector/state_of_art",
        "manual_tracking_with_yolov7/with_speed",
        "1",
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/metrics_results")
    MetricGoodDetections().compare_trackers_on_single_video(
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
    MetricGoodDetections().count_average_across_many_videos(
        "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/metrics_results",
        "pure_yolov7_detector/default",
        "pure_yolov7_detector/state_of_art",
        ["1", "2", "3"]
    )
