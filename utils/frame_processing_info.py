from utils.detection_result import Box


class FrameProcessingInfo:
    def __init__(self, frame_index: int, prediction_area: Box, detection_result: Box):
        self.frame_index = frame_index
        self.prediction_area = prediction_area
        self.detection_result = detection_result

    def to_dict(self):
        if self.detection_result is None:
            return {
                "frame_index": self.frame_index,
                "prediction_area": self.prediction_area.to_dict()
            }
        return {
            "frame_index": self.frame_index,
            "prediction_area": self.prediction_area.to_dict(),
            "detection_box": self.detection_result.to_dict()
        }

    @staticmethod
    def from_dict(data):
        if "detection_box" in data:
            return FrameProcessingInfo(int(data["frame_index"]),
                                       Box.from_dict(data["prediction_area"]),
                                       Box.from_dict(data["detection_box"]))
        return FrameProcessingInfo(int(data["frame_index"]), Box.from_dict(data["prediction_area"]), None)
