from utils.detection_result import Box


class FrameProcessingInfo:
    def __init__(self, frame_index: int, prediction_area: Box, detection_box: Box, estimate_box: Box, real_time_estimate_box: Box):
        self.frame_index = frame_index
        self.prediction_area = prediction_area
        self.detection_box = detection_box
        self.estimate_box = estimate_box
        self.real_time_estimate_box = real_time_estimate_box

    def to_dict(self):
        result_dict = {}
        if self.detection_box is not None:
            result_dict["detection_box"] = self.detection_box.to_dict()
        if self.estimate_box is not None:
            result_dict["estimate_box"] = self.estimate_box.to_dict()
        if self.prediction_area is not None:
            result_dict["prediction_area"] = self.prediction_area.to_dict()
        if self.real_time_estimate_box is not None:
            result_dict["real_time_estimate_box"] = self.real_time_estimate_box.to_dict()
        result_dict["frame_index"] = self.frame_index
        return result_dict

    @staticmethod
    def from_dict(data):
        detection_box = None
        prediction_area = None
        estimate_box = None
        real_time_estimate_box = None
        if "detection_box" in data:
            detection_box = Box.from_dict(data["detection_box"])
        if "prediction_area" in data:
            prediction_area = Box.from_dict(data["prediction_area"])
        if "estimate_box" in data:
            estimate_box = Box.from_dict(data["estimate_box"])
        if "real_time_estimate_box" in data:
            real_time_estimate_box = Box.from_dict(data["real_time_estimate_box"])
        frame_index = int(data["frame_index"])

        return FrameProcessingInfo(frame_index,
                                   prediction_area,
                                   detection_box,
                                   estimate_box,
                                   real_time_estimate_box)
