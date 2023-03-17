import sys
from pathlib import Path

import cv2

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.detection_result import Box


current_mouse_position = (20, 20)

def export_processed_video(raw_video_path: str,
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

    def on_click(event, x, y, p1, p2):
        global current_mouse_position

        if event == cv2.EVENT_MOUSEMOVE:
            cp = frame.copy()
            cv2.circle(cp, (x, y), 30, (255, 0, 0), 5)
            cv2.imshow("preview", cp)
            current_mouse_position = (x, y)

    cv2.namedWindow('preview')
    cv2.setMouseCallback('preview', on_click)


    current_frame_index = 0
    while status:
        # cv2.imshow("preview", frame)
        cv2.imshow("preview", frame)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        if key == ord(' '):
            # out_cap.write(frame)
            box = Box(current_mouse_position[0] / width, current_mouse_position[1] / height, 0.03, 0.03)
            print(box)
            # add box as detection result to raw file
        status, frame = in_cap.read()
        current_frame_index += 1

    out_cap.release()
    in_cap.release()

if __name__ == "__main__":
    export_processed_video("/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/1.mp4", "/home/kir/hawk-eye/HawkEyeExperiments/with_yolov7_detection/inference/1_output.mp4")