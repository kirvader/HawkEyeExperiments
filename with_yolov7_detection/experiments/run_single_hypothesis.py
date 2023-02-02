import cv2

from experiments.general_config import SECOND, PREDICTION_AREA_LINE_COLOR, PREDICTION_AREA_LINE_THICKNESS, \
    LAST_DETECTION_LINE_COLOR, LAST_DETECTION_LINE_THICKNESS, ESTIMATION_LINE_COLOR, ESTIMATION_LINE_THICKNESS
from tracker_base import SingleObjectTrackerBase


def run_hypothesis(tracker_impl: SingleObjectTrackerBase, input_source: str, output_source: str):
    in_cap = cv2.VideoCapture(input_source)
    width = in_cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
    height = in_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    fps = in_cap.get(cv2.CAP_PROP_FPS)
    out_cap = cv2.VideoWriter(output_source, -1, fps, (width, height))

    status, frame = in_cap.read()

    current_index = 0
    current_time = 0
    while status:
        if tracker_impl.is_available():
            prediction_area = tracker_impl.get_prediction_area(current_time)
            frame = prediction_area.draw(frame, width, height,
                                         PREDICTION_AREA_LINE_COLOR,
                                         PREDICTION_AREA_LINE_THICKNESS)

            last_detection = tracker_impl.process_frame(frame, current_time)

            if last_detection is not None:
                frame = last_detection.draw(frame, width, height,
                                            LAST_DETECTION_LINE_COLOR,
                                            LAST_DETECTION_LINE_THICKNESS)

        out_cap.write(frame)
        cv2.imshow("result", frame)

        status, frame = in_cap.read()
        current_index += 1
        current_time += SECOND / fps
