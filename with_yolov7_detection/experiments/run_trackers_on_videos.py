from argparse import ArgumentParser
from pathlib import Path

import sys

project_root = Path(__file__).parent.parent

sys.path.append(str(project_root))

from experiments.run_single_hypothesis import run_solution
from experiments.state_of_art_tracker.state_of_art_tracker import StateOfArtDetector
from experiments.yolov7_only_detection_tracker.yolov7_only_detection_tracker import YOLOv7OnlyDetectionTracker


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--videos_containing_folder', type=str, help="""
        This is a folder where all the VIDEO sources are stored.
    """)
    parser.add_argument('--video_names', nargs='+', default=[], help="""
        These are the NAMES(without format) of the videos in folder named videos_containing_folder.
        Should be in .mp4 format. 
    """)
    parser.add_argument('--results_directory', type=str, help="""
        This is a results directory where all the output will be stored.
        It will be created from scratch even if there are intermidiate folders.
    """)
    parser.add_argument('--trackers', nargs='+', default=[], help="""
        List of tracker names:
        - state_of_art_detector = YOLOv7 detection applied to each frame. Main trick here that this 
        - pure_yolov7_detector = YOLOv7 simple detection with "real" latency.
    """)
    parser.add_argument('--debug_mode', action='store_true', help="If turned on then it will show current video output")

    return parser.parse_args()


def get_tracker_by_name(tracker_name: str):
    if tracker_name == 'state_of_art_detector':
        return StateOfArtDetector()
    elif tracker_name == 'pure_yolov7_detector':
        return YOLOv7OnlyDetectionTracker()
    else:
        raise Exception("Provided tracker doesn't exist!")


if __name__ == "__main__":
    args = parse_args()

    results_directory = Path(args.results_directory)
    stored_videos_directory = Path(args.videos_containing_folder)
    for tracker_name in args.trackers:
        current_tracker_results_folder = results_directory / tracker_name
        current_tracker_results_folder.mkdir(parents=True, exist_ok=True)

        detector = get_tracker_by_name(tracker_name)
        for video_name in args.video_names:
            print(tracker_name, " ", video_name)
            current_video_path = str(results_directory / f"{video_name}.mp4")
            current_results_folder = current_tracker_results_folder / video_name
            current_results_folder.mkdir(parents=True, exist_ok=True)
            result_video_output_filename = str(current_results_folder / "visualization.mp4")
            result_raw_output_filename = str(current_results_folder / "raw.json")

            run_solution(detector,
                         current_video_path,
                         result_video_output_filename,
                         result_raw_output_filename,
                         args.debug_mode)
