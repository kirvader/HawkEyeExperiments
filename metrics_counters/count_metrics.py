import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from metrics_counters.metric_tracker_stability.metric_tracker_stability import MetricTrackerStability
from metrics_counters.metric_good_detections.metric_good_detections import MetricGoodDetections


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--results_path', type=str, help="""
        Path to results of inference which we are trying to rate.
    """)
    parser.add_argument('--videos', nargs='+', default=[], help="""
        Videos to rate trackers on.
    """)
    parser.add_argument('--state_of_art_tracker_name', type=str, help="""
        Results of good algorithm which we are trying to compare our solution with.
    """)
    parser.add_argument('--tracker_names', nargs='+', help="""
        Trackers which we will try to rate with state of art solution.
    """)
    parser.add_argument('--metrics', nargs='+', default=[], help="""
        Metric names with which we will try to rate our solution.
    """)

    return parser.parse_args()

def get_metrics_counter_by_name(metric_name: str):
    if metric_name == "detections_percentage":
        return MetricGoodDetections()
    if metric_name == "tracker_stability":
        return MetricTrackerStability()

if __name__ == "__main__":
    args = parse_args()
    state_of_art_results_path = Path(args.results_path) / args.state_of_art_tracker_name

    for metric_name in args.metrics:
        for tracker_name in args.tracker_names:
            tracker_results_path = Path(args.results_path) / tracker_name

            for video_name in args.videos:
                state_of_art_tracker_results_for_video = state_of_art_results_path / video_name / 'raw.json'
                current_tracker_results_for_video = tracker_results_path / video_name / 'raw.json'

                metric_counter = get_metrics_counter_by_name(metric_name)
                metric_counter.count(str(current_tracker_results_for_video), str(state_of_art_tracker_results_for_video))

        metric_counter = get_metrics_counter_by_name(metric_name)
        metric_counter.plot_all_on_one(args.results_path, args.tracker_names, args.videos)