class Box:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

def transform_to_absolute_from_relative(relative_box, box_container: Box):
    relative_box.x = box_container.x - box_container.w / 2 + box_container.w * relative_box.x
    relative_box.y = box_container.y - box_container.h / 2 + box_container.h * relative_box.y

    relative_box.w *= box_container.w
    relative_box.h *= box_container.h

class DetectionResult:
    def __init__(self, x, y, w, h, cls, conf = 1.0):
        self.box = Box(x, y, w, h)
        self.conf = conf
        self.cls = cls
