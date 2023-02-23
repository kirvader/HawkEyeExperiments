import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent

sys.path.append(str(project_root))

from experiments.inference_utils.detection_result import Box


def get_cropped_image(frame, part_to_keep: Box):
    top = max(0, int(frame.shape[0] * (part_to_keep.y - part_to_keep.h / 2)))
    left = max(0, int(frame.shape[1] * (part_to_keep.x - part_to_keep.w / 2)))
    bottom = min(frame.shape[0] - 1, int(frame.shape[0] * (part_to_keep.y + part_to_keep.h / 2)))
    right = min(frame.shape[1] - 1, int(frame.shape[1] * (part_to_keep.x + part_to_keep.w / 2)))

    if bottom - top < 32:
        if top + 32 < frame.shape[0]:
            bottom = top + 32
        else:
            top = bottom - 32
    if right - left < 32:
        if left + 32 < frame.shape[0]:
            right = left + 32
        else:
            left = right - 32

    return frame[top:bottom, left:right].copy()