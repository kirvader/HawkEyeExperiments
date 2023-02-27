import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from metrics_counters.metric_tracker_stability.metric_tracker_stability_whole_bitmap import MetricTrackerStabilityWholeBitmap
from metrics_counters.metric_tracker_stability.metric_tracker_stability_object_in_scope_time import MetricTrackerStabilityObjectInScopeTime
from metrics_counters.metric_good_detections.metric_good_detections import MetricGoodDetections


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--raw_results_path', type=str, help="""
        Path to raw results of inference which we are trying to rate.
    """)
    parser.add_argument('--metrics_results_path', type=str, help="""
        Path to where to store the metrics results
    """)
    parser.add_argument('--videos', nargs='+', default=[], help="""
        Videos to rate trackers on.
    """)
    parser.add_argument('--state_of_art_tracker_with_config', type=str, help="""
        Name of tracker with config of the state of art tracker in format "<tracker_name>/<config_name>"
    """)
    parser.add_argument('--tracker_names_with_config', nargs='+', help="""
        Trackers which we will try to rate with state of art solution.
        - pure_yolov7_detector = YOLOv7 simple detection with "real" latency. Configs:
          * "default" config - the latency of inference is emulated.
          * "state_of_art" - the best detector without latency.
        - manual_tracking_with_yolov7 = Manual tracking including object speed control, detection via YOLOv7. Idea is to find the object in the area it was found on previous frame according to object speed.
          * no_speed - no dependency of object speed.
          * with_speed - estimation depends on current speed of the object.
    """)
    parser.add_argument('--metrics', nargs='+', default=[], help="""
        Metric names with which we will try to rate our solution.
        - detections_percentage = YOLOv7 detection applied to each frame. Main trick here that this 
        - tracker_stability_whole_bitmap = YOLOv7 simple detection with "real" latency.
        - tracker_stability_object_in_scope = Manual tracking including object speed control, detection via YOLOv7. Idea is to find the object in the area it was found on previous frame according to object speed.   
    """)
    parser.add_argument('--max_columns', type=int, default=4, help="""
        When printing all charts on one, max_columns will be used here
    """)

    return parser.parse_args()

def get_metrics_counter_by_name(metric_name: str):
    if metric_name == "detections_percentage":
        return MetricGoodDetections()
    if metric_name == "tracker_stability_whole_bitmap":
        return MetricTrackerStabilityWholeBitmap()
    if metric_name == "tracker_stability_object_in_scope":
        return MetricTrackerStabilityObjectInScopeTime()

if __name__ == "__main__":
    args = parse_args()
    state_of_art_results_path = Path(args.results_path) / args.state_of_art_tracker_with_config

    for metric_name in args.metrics:
        for video_name in args.videos:
            for tracker_with_config in args.trackers_with_configs:
                tracker_results_path = Path(args.results_path) / tracker_with_config
                state_of_art_tracker_results_for_video = state_of_art_results_path / video_name / 'raw.json'
                current_tracker_results_for_video = tracker_results_path / video_name / 'raw.json'
                metric_counter = get_metrics_counter_by_name(metric_name)
                metric_counter.count(str(current_tracker_results_for_video), str(state_of_art_tracker_results_for_video))
        metric_counter = get_metrics_counter_by_name(metric_name)
        metric_counter.plot_all_on_one(args.results_path, args.tracker_names, args.videos, args.max_columns)
