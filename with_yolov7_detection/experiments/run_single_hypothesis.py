import json
from pathlib import Path

import cv2

from experiments.general_config import SECOND, PREDICTION_AREA_LINE_COLOR, PREDICTION_AREA_LINE_THICKNESS, \
    LAST_DETECTION_LINE_COLOR, LAST_DETECTION_LINE_THICKNESS, ESTIMATION_LINE_COLOR, ESTIMATION_LINE_THICKNESS
from experiments.inference_utils.frame_processing_info import FrameProcessingInfo
from experiments.tracker_base import SingleObjectTrackerBase


def run_solution(tracker_impl: SingleObjectTrackerBase,
                 video_input_source: str,
                 video_output_source: str,
                 json_results_path: str,
                 debug_mode=False):
    in_cap = cv2.VideoCapture(video_input_source)
    if in_cap is None or not in_cap.isOpened():
        print(f"Cannot open provided video source \"{video_input_source}\"")
        return
    width = int(in_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
    height = int(in_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    fps = int(in_cap.get(cv2.CAP_PROP_FPS))
    out_cap = cv2.VideoWriter(video_output_source, cv2.VideoWriter_fourcc(*"DIVX"), fps, (width, height))

    status, frame = in_cap.read()
    raw_results_file = open(json_results_path, "w")
    raw_results_file.write("[\n")

    current_index = 0
    current_time = 0
    is_first_detection = True
    while status:
        if tracker_impl.is_available(current_time):
            prediction_area = tracker_impl.get_prediction_area(current_time)
            frame = prediction_area.draw(frame, width, height,
                                         PREDICTION_AREA_LINE_COLOR,
                                         PREDICTION_AREA_LINE_THICKNESS)

            last_detection = tracker_impl.process_frame(frame, current_time)

            if last_detection is not None:
                frame = last_detection.draw(frame, width, height,
                                            LAST_DETECTION_LINE_COLOR,
                                            LAST_DETECTION_LINE_THICKNESS)
            if is_first_detection:
                is_first_detection = False
            else:
                raw_results_file.write(',\n')
            raw_results_file.write(
                json.dumps(FrameProcessingInfo(current_index, prediction_area, last_detection).to_dict(), indent=4))

        out_cap.write(frame)
        if debug_mode:
            cv2.imshow("result", frame)
        status, frame = in_cap.read()
        current_index += 1
        current_time += SECOND / fps

    out_cap.release()
    in_cap.release()
    raw_results_file.write("]")
    raw_results_file.close()

