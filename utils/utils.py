import argparse
import os


def parse():
    parser = argparse.ArgumentParser(description="ddn ba")

    parser.add_argument(
        "-s", "--scene", type=str, help="scene to generate", default="fire"
    )

    parser.add_argument(
        "-l", "--length", type=int, help="sequence length", default="1000"
    )

    return parser.parse_args()
