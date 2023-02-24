import json
from argparse import ArgumentParser
from pathlib import Path

import cv2
from tqdm import tqdm

from box_drawer_config import PREDICTION_AREA_LINE_COLOR, \
    PREDICTION_AREA_LINE_THICKNESS, \
    LAST_DETECTION_LINE_COLOR, \
    LAST_DETECTION_LINE_THICKNESS
from utils.frame_processing_info import FrameProcessingInfo


def apply_processing_result_to_frame(frame, frame_processing_info, width, height):
    if frame_processing_info.prediction_area is not None:
        frame = frame_processing_info.prediction_area.draw(frame, width, height,
                                                           PREDICTION_AREA_LINE_COLOR,
                                                           PREDICTION_AREA_LINE_THICKNESS)
    if frame_processing_info.detection_result is not None:
        frame = frame_processing_info.detection_result.draw(frame, width, height,
                                                            LAST_DETECTION_LINE_COLOR,
                                                            LAST_DETECTION_LINE_THICKNESS)


def concat_frames(frames: list):
    if len(frames) == 1:
        return frames[0]
    if len(frames) == 2 or len(frames) == 3:
        return cv2.hconcat(frames)
    if len(frames) == 4:
        return cv2.vconcat([cv2.hconcat(frames[:2]), cv2.hconcat(frames[2:])])


def export_processed_video(raw_video_path: str,
                           raw_results_path_and_name: list,
                           output_video_path: str):
    in_cap = cv2.VideoCapture(raw_video_path)
    if in_cap is None or not in_cap.isOpened():
        print(f"Cannot open provided video source \"{raw_video_path}\"")
        return
    width = int(in_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
    height = int(in_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    fps = int(in_cap.get(cv2.CAP_PROP_FPS))
    out_cap = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"DIVX"), fps, (2 * width, 2 * height))

    status, frame = in_cap.read()
    tracker_names = []
    raw_results = []
    for raw_result_path, tracker_config_name in raw_results_path_and_name:
        raw_results_file = open(raw_result_path)
        results_data = list(
            map(lambda item: FrameProcessingInfo.from_dict(item), json.load(raw_results_file)))
        raw_results_file.close()
        raw_results.append(results_data)
        tracker_names.append(tracker_config_name)

    current_frame_index = 0
    current_results_index = [0 for _ in range(len(raw_results))]
    while status:
        current_frame_visualizations = []
        for i in range(len(raw_results)):
            frame_i = frame.copy()
            if current_results_index[i] != len(raw_results[i]):
                if current_frame_index == raw_results[i][current_results_index[i]].frame_index:
                    apply_processing_result_to_frame(frame, raw_results[i][current_results_index[i]], width, height)
                    current_results_index += 1
            frame_i = cv2.putText(frame_i, tracker_names[i], (0, 0), cv2.FONT_HERSHEY_SIMPLEX,
                                  1, (255, 255, 255), 2, cv2.LINE_AA)
            current_frame_visualizations.append(frame_i)

        out_cap.write(concat_frames(current_frame_visualizations))
        status, frame = in_cap.read()
        current_frame_index += 1

    out_cap.release()
    in_cap.release()


def export_processed_videos_one_by_one(raw_results_directory: str,
                                       tracker_names_with_config: list,
                                       raw_videos_directory: str,
                                       video_names: list,
                                       output_directory: str):
    for video_name in tqdm(video_names, position=0, leave=True):
        current_raw_video_path = Path(raw_videos_directory) / f"{video_name}.mp4"
        for tracker_with_config in tqdm(tracker_names_with_config, position=1, leave=True):
            raw_results_path = Path(raw_results_directory) / tracker_with_config / video_name / "raw.json"
            tracker_config_split = tracker_with_config.split("/")
            tracker_name = tracker_config_split[0]
            config_name = tracker_config_split[1]
            output_video_path = Path(output_directory) / f"{video_name}:{tracker_name}:{config_name}.mp4"
            export_processed_video(str(current_raw_video_path), [(str(raw_results_path), str(raw_results_path))],
                                   str(output_video_path))


def export_processed_videos_all_in_one_video(raw_results_directory: str,
                                             tracker_names_with_config: list,
                                             raw_videos_directory: str,
                                             video_names: list,
                                             output_directory: str):
    assert (len(tracker_names_with_config) <= 4)
    for video_name in video_names:
        current_raw_video_path = Path(raw_videos_directory) / f"{video_name}.mp4"
        raw_results_path_and_name = []
        output_filename = f"{video_name}"
        for tracker_with_config in tqdm(tracker_names_with_config, position=1, leave=True):
            raw_results_path = Path(raw_results_directory) / tracker_with_config / video_name / "raw.json"
            raw_results_path_and_name.append((str(raw_results_path), tracker_with_config))
            tracker_config_split = tracker_with_config.split("/")
            tracker_name = tracker_config_split[0]
            config_name = tracker_config_split[1]
            output_filename += f"-{tracker_name}:{config_name}"
        output_filename += ".mp4"
        export_processed_video(str(current_raw_video_path), raw_results_path_and_name,
                               str(Path(output_directory) / output_filename))


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
    parser.add_argument('--tracker_names_with_config', nargs='+', default=[], help="""
        List of tracker names with config like "<tracker_name>/<config_name>". Tracker names:
        - state_of_art_detector = YOLOv7 detection applied to each frame. Main trick here that this 
        - pure_yolov7_detector = YOLOv7 simple detection with "real" latency.
        - manual_tracking_with_yolov7 = Manual tracking including object speed control, detection via YOLOv7. Idea is to find the object in the area it was found on previous frame according to object speed.
    """)
    parser.add_argument('--output_directory', type=str, help="Results will be stored here!")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_processed_videos_all_in_one_video(args.results_directory,
                                             args.tracker_names_with_config,
                                             args.videos_containing_folder,
                                             args.video_names,
                                             args.output_directory)
