import sys
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm_notebook as tqdm

sys.path.append(str(Path(__file__).parent.parent))

from metrics_counters.metric_tracker_stability.metric_tracker_stability_object_in_scope_time import \
    MetricTrackerStabilityObjectInScopeTime
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
        - tracker_stability_object_in_scope = Manual tracking including object speed control, detection via YOLOv7. Idea is to find the object in the area it was found on previous frame according to object speed.   
    """)

    return parser.parse_args()


def get_metrics_counter_by_name(metric_name: str):
    if metric_name == "detections_percentage":
        return MetricGoodDetections()
    if metric_name == "tracker_stability_object_in_scope":
        return MetricTrackerStabilityObjectInScopeTime()


if __name__ == "__main__":
    args = parse_args()

    metrics_pbar = tqdm(args.metrics, position=0, leave=False)
    for metric_name in metrics_pbar:
        metrics_pbar.set_description(metric_name)
        metrics_pbar.set_postfix_str("Single tracker - single video")


        trackers_pbar = tqdm(args.tracker_names_with_config, position=1, leave=False)
        for tracker_with_config in trackers_pbar:
            trackers_pbar.set_description(f"Tracker {tracker_with_config}")

            videos_pbar = tqdm(args.videos, position=2, leave=False)
            for video_name in videos_pbar:
                videos_pbar.set_description(f"Video \"{video_name}\"")

                metric_counter = get_metrics_counter_by_name(metric_name)
                metric_counter.count_tracker_performance_on_single_video(
                    args.raw_results_path,
                    args.state_of_art_tracker_with_config,
                    tracker_with_config,
                    video_name,
                    args.metrics_results_path
                )

        metrics_pbar.set_postfix_str("Comparing trackers on each video")
        videos_pbar = tqdm(args.videos, position=1, leave=False)
        for video_name in videos_pbar:
            videos_pbar.set_description(f"Video \"{video_name}\"")
            metric_counter = get_metrics_counter_by_name(metric_name)
            metric_counter.compare_trackers_on_single_video(
                args.metrics_results_path,
                args.tracker_names_with_config,
                args.state_of_art_tracker_with_config,
                video_name
            )

        metrics_pbar.set_postfix_str("Averaging each tracker over all videos")
        trackers_pbar = tqdm(args.tracker_names_with_config, position=1, leave=False)
        for tracker_with_config in trackers_pbar:
            trackers_pbar.set_description(f"Tracker {tracker_with_config}")
            metric_counter = get_metrics_counter_by_name(metric_name)
            metric_counter.count_average_across_many_videos(
                args.metrics_results_path,
                tracker_with_config,
                args.state_of_art_tracker_with_config,
                args.videos
            )
    print("Finished! \nProcessed parameters.\n")
    print("Metrics\n\t", "\n\t".join(args.metrics))
    print("Trackers\n\t", "\n\t".join(args.tracker_names_with_config))
    print("Videos\n\t", "\n\t".join(args.videos))
