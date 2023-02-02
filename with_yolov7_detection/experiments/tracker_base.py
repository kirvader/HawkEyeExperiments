from experiments.inference_utils.detection_result import Box


class SingleObjectTrackerBase:
    def is_available(self, timestamp: int) -> bool:
        pass

    def process_frame(self, frame, timestamp: int) -> Box:
        pass

    def get_prediction_area(self, timestamp: int) -> Box:
        pass
