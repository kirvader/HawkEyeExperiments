import json

import cv2


from experiments.inference_utils.frame_processing_info import FrameProcessingInfo
from experiments.tracker_base import SingleObjectTrackerBase

from tqdm import tqdm

def run_solution(tracker_impl: SingleObjectTrackerBase,
                 video_input_source: str,
                 json_results_path: str,
                 debug_mode=False):
    in_cap = cv2.VideoCapture(video_input_source)
    if in_cap is None or not in_cap.isOpened():
        print(f"Cannot open provided video source \"{video_input_source}\"")
        return
    fps = int(in_cap.get(cv2.CAP_PROP_FPS))
    length = int(in_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    status, frame = in_cap.read()
    raw_results_file = open(json_results_path, "w")
    raw_results_file.write("[\n")

    current_index = 0
    current_time = 0
    is_first_detection = True
    with tqdm(range(1, length), desc=f"{tracker_impl.__class__.__name__}", position=0, leave=True) as tqdm_bar:
        for _ in tqdm_bar:
            if tracker_impl.is_available(current_time):
                prediction_area = tracker_impl.get_prediction_area(current_time)

                last_detection = tracker_impl.process_frame(frame, current_time)

                if last_detection is None:
                    tqdm_bar.set_postfix_str(f"{current_index} - None")
                else:
                    tqdm_bar.set_postfix_str(f"{current_index} - Found")
                tqdm_bar.refresh()

                if is_first_detection:
                    is_first_detection = False
                else:
                    raw_results_file.write(',\n')
                raw_results_file.write(
                    json.dumps(FrameProcessingInfo(current_index, prediction_area, last_detection).to_dict(), indent=4))

            if debug_mode:
                cv2.imshow("result", frame)
            status, frame = in_cap.read()
            current_index += 1
            current_time += 1000 / fps  # ms_in_second / fps

    in_cap.release()
    raw_results_file.write("]")
    raw_results_file.close()

