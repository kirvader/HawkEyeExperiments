from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm as tqdm
# from tqdm import tqdm_notebook as tqdm

import sys

project_root = Path(__file__).parent.parent

sys.path.append(str(project_root))

from experiments.run_single_hypothesis import run_solution
from experiments.state_of_art_tracker.state_of_art_tracker import StateOfArtDetector
from experiments.yolov7_only_detection_tracker.yolov7_only_detection_tracker import YOLOv7OnlyDetectionTracker
from experiments.yolov7_manual_tracker.yolov7_manual_tracker import YOLOv7ManualTracker
from experiments.yolov7_manual_tracker.yolov7_manual_tracker_no_speed_control import \
    YOLOv7ManualTrackerNoSpeedControl


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
        It will be created from scratch even if there are intermediate folders.
    """)
    parser.add_argument('--trackers', nargs='+', default=[], help="""
        List of tracker names:
        - state_of_art_detector = YOLOv7 detection applied to each frame. Main trick here that this 
        - pure_yolov7_detector = YOLOv7 simple detection with "real" latency.
        - manual_tracking_with_yolov7 = Manual tracking including object speed control, detection via YOLOv7. Idea is to find the object in the area it was found on previous frame according to object speed.
        - manual_tracking_with_yolov7 = Manual tracking without object speed control, detection via YOLOv7. Idea is to find the object in the area it was found on previous frame.
    """)
    parser.add_argument('--debug_mode', action='store_true', help="If turned on then it will show current video output")

    return parser.parse_args()


def get_tracker_by_name(tracker_name: str):
    if tracker_name == 'state_of_art_detector':
        return StateOfArtDetector()
    elif tracker_name == 'pure_yolov7_detector':
        return YOLOv7OnlyDetectionTracker()
    elif tracker_name == 'manual_tracking_with_yolov7':
        return YOLOv7ManualTracker()
    elif tracker_name == 'manual_tracking_with_yolov7_no_speed_control':
        return YOLOv7ManualTrackerNoSpeedControl()
    else:
        raise Exception("Provided tracker doesn't exist!")


if __name__ == "__main__":
    args = parse_args()

    results_directory = Path(args.results_directory)
    stored_videos_directory = Path(args.videos_containing_folder)
    with tqdm(args.trackers) as tracker_pbar:
        for tracker_name in tracker_pbar:
            tracker_pbar.set_description(f"Tracker {tracker_name}")
            current_tracker_results_folder = results_directory / tracker_name
            current_tracker_results_folder.mkdir(parents=True, exist_ok=True)
            with tqdm(args.video_names) as videos_for_tracker_pbar:
                for video_name in videos_for_tracker_pbar:
                    videos_for_tracker_pbar.set_description(f"Video {video_name}")
                    current_video_path = str(stored_videos_directory / f"{video_name}.mp4")
                    current_results_folder = current_tracker_results_folder / video_name
                    current_results_folder.mkdir(parents=True, exist_ok=True)
                    result_video_output_filename = str(current_results_folder / "visualization.mp4")
                    result_raw_output_filename = str(current_results_folder / "raw.json")
                    detector = get_tracker_by_name(tracker_name)

                    run_solution(detector,
                                 current_video_path,
                                 result_raw_output_filename,
                                 args.debug_mode)
