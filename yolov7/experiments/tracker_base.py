from utils.detection_result import Box


class SingleObjectTrackerBase:
    def init(self):
        pass

    def process_frame_if_possible(self):
        pass

    def get_estimate(self) -> Box:
        pass

    def get_prediction_area(self) -> Box:
        pass

    def get_detection(self) -> Box:
        pass