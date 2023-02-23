import json
from pathlib import Path

import cv2

from box_drawer_config import PREDICTION_AREA_LINE_COLOR, \
                              PREDICTION_AREA_LINE_THICKNESS, \
                              LAST_DETECTION_LINE_COLOR, \
                              LAST_DETECTION_LINE_THICKNESS
from utils.frame_processing_info import FrameProcessingInfo


def export_processed_video(raw_video_path: str,
                           raw_result_path: str,
                           output_video_path: str):
    in_cap = cv2.VideoCapture(raw_video_path)
    if in_cap is None or not in_cap.isOpened():
        print(f"Cannot open provided video source \"{raw_video_path}\"")
        return
    width = int(in_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
    height = int(in_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    fps = int(in_cap.get(cv2.CAP_PROP_FPS))
    out_cap = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"DIVX"), fps, (width, height))

    status, frame = in_cap.read()
    raw_results_file = open(raw_result_path)
    results_data = list(
        map(lambda item: FrameProcessingInfo.from_dict(item), json.load(raw_results_file)))
    raw_results_file.close()

    current_frame_index = 0
    current_results_index = 0
    while status:
        if current_results_index == len(results_data):
            out_cap.write(frame)
            status, frame = in_cap.read()
            continue
        if current_frame_index == results_data[current_results_index].frame_index:
            if results_data[current_results_index].prediction_area is not None:
                frame = results_data[current_results_index].prediction_area.draw(frame, width, height,
                                                                                 PREDICTION_AREA_LINE_COLOR,
                                                                                 PREDICTION_AREA_LINE_THICKNESS)
            if results_data[current_results_index].detection_result is not None:
                frame = results_data[current_results_index].detection_result.draw(frame, width, height,
                                                                                  LAST_DETECTION_LINE_COLOR,
                                                                                  LAST_DETECTION_LINE_THICKNESS)
            current_results_index += 1

        out_cap.write(frame)
        status, frame = in_cap.read()
        current_frame_index += 1

    out_cap.release()
    in_cap.release()
