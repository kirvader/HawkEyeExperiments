import json
from pathlib import Path

from utils.frame_processing_info import FrameProcessingInfo


class MetricCounterBase:
    def compare_trackers_on_single_video(self,
                                         metrics_output_directory: str,
                                         tracker_names_with_configs: list,
                                         state_of_art_tracker_with_config: str,
                                         video_name: str):
        pass

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
        pass

    @staticmethod
    def get_output_folder_for_single_tracker_single_video(state_of_art_tracker_with_config: str,
                                                          current_tracker_with_config: str,
                                                          video_name: str,
                                                          metrics_output_directory: str,
                                                          metric_name: str) -> Path:
        current_comparison_output_directory = Path(
            metrics_output_directory) / "by_tracker" / current_tracker_with_config / f"{metric_name}:{state_of_art_tracker_with_config.replace('/', ':')}/" / video_name
        current_comparison_output_directory.mkdir(parents=True, exist_ok=True)
        return current_comparison_output_directory

    @staticmethod
    def get_output_folder_for_single_tracker_many_video(state_of_art_tracker_with_config: str,
                                                        current_tracker_with_config: str,
                                                        metrics_output_directory: str,
                                                        metric_name: str) -> Path:
        current_comparison_output_directory = Path(
            metrics_output_directory) / "by_tracker" / current_tracker_with_config / f"{metric_name}:{state_of_art_tracker_with_config.replace('/', ':')}/"
        current_comparison_output_directory.mkdir(parents=True, exist_ok=True)
        return current_comparison_output_directory

    @staticmethod
    def get_output_folder_for_many_trackers_single_video(state_of_art_tracker_with_config: str,
                                                         trackers_name_with_config: list,
                                                         video_name: str,
                                                         metrics_output_directory: str,
                                                         metric_name: str) -> Path:
        comparing_list_of_trackers = "-".join(map(lambda s: s.replace('/', ':'), trackers_name_with_config))
        current_comparison_output_directory = Path(
            metrics_output_directory) / "by_video" / video_name / f"{metric_name}:{state_of_art_tracker_with_config.replace('/', ':')}" / comparing_list_of_trackers
        current_comparison_output_directory.mkdir(parents=True, exist_ok=True)
        return current_comparison_output_directory

    @staticmethod
    def read_raw_results(raw_results_directory: str,
                         tracker_with_config: str,
                         video_name: str) -> list:
        raw_considering_results_filename = str(
            Path(raw_results_directory) / tracker_with_config / video_name / "raw.json")
        with open(raw_considering_results_filename) as considering_results_file:
            return list(
                map(lambda item: FrameProcessingInfo.from_dict(item), json.load(considering_results_file)))
