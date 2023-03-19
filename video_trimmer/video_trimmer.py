import json
import sys
from pathlib import Path

import cv2


project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.frame_processing_info import FrameProcessingInfo
from utils.detection_result import Box


current_mouse_position = (20, 20)

def export_processed_video(raw_video_path: str,
                           raw_results_path: str,
                           output_video_path: str):
    in_cap = cv2.VideoCapture(raw_video_path)
    if in_cap is None or not in_cap.isOpened():
        print(f"Cannot open provided video source \"{raw_video_path}\"")
        return
    width = int(in_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
    height = int(in_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    fps = int(in_cap.get(cv2.CAP_PROP_FPS))
    Path(raw_results_path).parent.mkdir(parents=True, exist_ok=True)

    out_cap = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"DIVX"), fps, (width, height))

    status, frame = in_cap.read()

    def on_click(event, x, y, p1, p2):
        global current_mouse_position

        if event == cv2.EVENT_MOUSEMOVE:
            cp = frame.copy()
            cv2.circle(cp, (x, y), 30, (255, 0, 0), 5)
            cv2.imshow("preview", cp)
            current_mouse_position = (x, y)

    cv2.namedWindow('preview')
    cv2.setMouseCallback('preview', on_click)

    raw_results_file = open(raw_results_path, "w")
    raw_results_file.write("[\n")

    current_frame_index = 0
    written_frame_index = 0
    is_first_line = True
    while status:
        # cv2.imshow("preview", frame)
        cv2.imshow("preview", frame)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        if key == ord(' '):
            out_cap.write(frame)
            box = Box(current_mouse_position[0] / width, current_mouse_position[1] / height, 0.03, 0.03)
            print(box)
            # add box as detection result to raw file
            if is_first_line:
                is_first_line = False
            else:
                raw_results_file.write(",\n")
            raw_results_file.write(json.dumps(FrameProcessingInfo(written_frame_index, None, box, None,
                                           None).to_dict(), indent=4))
            written_frame_index += 1
        status, frame = in_cap.read()
        current_frame_index += 1
    raw_results_file.write("\n]")
    out_cap.release()
    in_cap.release()

if __name__ == "__main__":
    parent_path = "/home/kir/hawk-eye/HawkEyeExperiments/video_trimmer"
    for i in [5]:
        export_processed_video(f"{parent_path}/videos/{i}.mp4",
                               f"{parent_path}/raw_results/{i}/raw.json",
                               f"{parent_path}/trimmed_videos/{i}.mp4")