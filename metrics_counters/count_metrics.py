from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--considering_results_filename', type=str, help="""
        Results of inference which we are trying to rate.
    """)
    parser.add_argument('--state_of_art_results_filename', type=str, help="""
        Results of good algorithm which we are trying to compare our solution with.
    """)
    parser.add_argument('--metrics', nargs='+', default=[], help="""
        Metric names with which we will try to rate our solution
    """)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
