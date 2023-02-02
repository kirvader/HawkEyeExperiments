from detection_result import Box


class FrameProcessingInfo:
    def __init__(self, prediction_area: Box, detection_result: Box):
        self.prediction_area = prediction_area
        self.detection_result = detection_result

    def to_dict(self):
        if self.detection_result is None:
            return {
                "prediction_area": self.prediction_area.to_dict()
            }
        return {
            "prediction_area": self.prediction_area.to_dict(),
            "detection_box": self.detection_result.to_dict()
        }

    @staticmethod
    def from_dict(data):
        if "detection_box" in data:
            return FrameProcessingInfo(Box.from_dict(data["prediction_area"]),
                                       Box.from_dict(data["detection_box"]))
        return FrameProcessingInfo(Box.from_dict(data["prediction_area"]), Box(0, 0, 0, 0))

