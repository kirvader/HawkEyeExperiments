import cv2


class Box:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __str__(self):
        return f"{self.x} {self.y} {self.w} {self.w}"

    def to_dict(self):
        return {
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h
        }

    @staticmethod
    def from_dict(data):
        return Box(float(data["x"]), float(data["y"]), float(data["w"]), float(data["h"]))

    def draw(self, frame, width, height, color, thickness):
        left = max(0, int(width * (self.x - self.w / 2)))
        right = min(width - 1, int(width * (self.x + self.w / 2)))
        top = max(0, int(height * (self.y - self.h / 2)))
        bottom = min(height - 1, int(height * (self.y + self.h / 2)))

        return cv2.rectangle(frame,
                             (left, top),
                             (right, bottom),
                             color,
                             thickness)

    def is_close_to(self, other, eps: float):
        return abs(self.x - other.x) < eps and abs(self.y - other.y) < eps

    def square_of_intersection(self, other):
        left = max(self.x - self.w / 2, other.x - other.w / 2)
        top = max(self.y - self.h / 2, other.y - other.h / 2)
        right = min(self.x + self.w / 2, other.x + other.w / 2)
        bottom = max(self.y + self.h / 2, other.y + other.h / 2)

        return max(0.0, (bottom - top) * (right - left))

    def square_of_union(self, other):
        return self.w * self.h + other.w * other.h - self.square_of_intersection(other)


def transform_to_absolute_from_relative(relative_box: Box, box_container: Box) -> Box:
    return Box(box_container.x - box_container.w / 2 + box_container.w * relative_box.x,
        box_container.y - box_container.h / 2 + box_container.h * relative_box.y,
        relative_box.w * box_container.w,
        relative_box.h * box_container.h)


class DetectionResult:
    def __init__(self, x, y, w, h, cls, conf=1.0):
        self.box = Box(x, y, w, h)
        self.conf = conf
        self.cls = cls

    def __str__(self):
        return f"{str(self.box)}, cls = {self.cls}, conf = {self.conf}"
