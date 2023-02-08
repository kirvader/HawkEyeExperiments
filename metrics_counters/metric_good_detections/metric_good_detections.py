import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
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


def plot_single_horizontal_chart(videos, fst_part, snd_part):
    frames_handled = [fst_part]
    all_frames_count = [snd_part]

    b1 = plt.barh(videos, frames_handled, color="blue")

    b2 = plt.barh(videos, all_frames_count, left=frames_handled, color="gray")

    plt.legend([b1, b2], ["Handled", "Not handled"], title="Efficiency", loc="upper right")


class DetectionResults:
    def __init__(self, frames_quantity=0, detections_run=0, positive_true=0, positive_false=0, negative_true=0,
                 negative_false=0, comparing_file=""):
        self.frames_quantity = frames_quantity
        self.detections_run = detections_run
        self.positive_true = positive_true
        self.positive_false = positive_false
        self.negative_true = negative_true
        self.negative_false = negative_false
        self.comparing_file = comparing_file

    def take_into_account(self, other):
        self.frames_quantity += other.frames_quantity
        self.detections_run += other.detections_run
        self.positive_true += other.positive_true
        self.positive_false += other.positive_false
        self.negative_true += other.negative_true
        self.negative_false += other.negative_false

    def to_dict(self):
        return {
            "frames_quantity": self.frames_quantity,
            "detections_run": self.detections_run,
            "positive_true": self.positive_true,
            "positive_false": self.positive_false,
            "negative_true": self.negative_true,
            "negative_false": self.negative_false,
            "comparing_file": self.comparing_file
        }

    def plot(self, filename):
        figure = plt.figure(figsize=(25, 25))
        figure.add_subplot(2, 1, 2)
        plot_single_horizontal_chart([""], self.detections_run, self.frames_quantity - self.detections_run)

        figure.add_subplot(2, 1, 1)
        DetectionCounter(self.positive_true, self.positive_false, self.negative_true, self.negative_false).plot()

        plt.savefig(filename)
        plt.close()

    @staticmethod
    def from_dict(data):
        return DetectionResults(data["frames_quantity"], data["frames_quantity"], data["positive_true"],
                                data["positive_false"], data["negative_true"], data["negative_false"],
                                data["comparing_file"])


class MetricGoodDetections(MetricCounterBase):
    METRIC_NAME = "METRIC_DETECTIONS_PERCENTAGE"

    def plot_all_on_one(self, results_folder, tracker_names, video_names):
        columns = len(video_names)
        rows = 2 * len(tracker_names)
        result_figure = plt.figure(figsize=(25, 25))

        for tracker_index in range(len(tracker_names)):
            path = Path(results_folder) / tracker_names[tracker_index]
            overall_results = DetectionResults()
            for video_index in range(len(video_names)):
                with open(path / video_names[video_index] / f"{MetricGoodDetections.METRIC_NAME}.json") as f:
                    overall_results.take_into_account(DetectionResults.from_dict(json.load(f)))

                index_in_grid = tracker_index * 2 * columns + video_index + 1
                filename = path / video_names[
                    video_index] / f"{MetricGoodDetections.METRIC_NAME}_true_false_positive_negative.png"
                img = mpimg.imread(str(filename))
                result_figure.add_subplot(rows, columns, index_in_grid)
                plt.imshow(img)

                index_in_grid = tracker_index * 2 * columns + columns + video_index + 1
                filename = path / video_names[video_index] / f"{MetricGoodDetections.METRIC_NAME}_efficiency.png"
                img = mpimg.imread(str(filename))
                result_figure.add_subplot(rows, columns, index_in_grid)
                plt.imshow(img)
            plt.savefig(str(Path(results_folder) / tracker_names[tracker_index] / f"{MetricGoodDetections.METRIC_NAME}_all_charts_on_one.pdf"))
            plt.close()
            overall_results.plot(str(Path(results_folder) / tracker_names[tracker_index] / f"{MetricGoodDetections.METRIC_NAME}_all_in_one.pdf"))

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

        with open(Path(raw_considering_results_filename).parent / f"{MetricGoodDetections.METRIC_NAME}.json",
                  "w") as json_file:
            json_file.write(json.dumps(
                DetectionResults(frames_quantity, len(considering_data), detections_counter.detected_right,
                                 detections_counter.detected_wrong, detections_counter.not_detected_right,
                                 detections_counter.not_detected_wrong, raw_state_of_art_results_filename).to_dict(),
                indent=4))

        plot_single_horizontal_chart([""], len(considering_data), frames_quantity - len(considering_data))
        plt.savefig(str(Path(
            raw_considering_results_filename).parent / f"{MetricGoodDetections.METRIC_NAME}_efficiency.png"))
        plt.close()

        detections_counter.plot()
        plt.savefig(str(Path(
            raw_considering_results_filename).parent / f"{MetricGoodDetections.METRIC_NAME}_true_false_positive_negative.png"))
        plt.close()


if __name__ == "__main__":
    MetricGoodDetections().count("/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/1_raw.json",
                                 "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/1_raw.json")
