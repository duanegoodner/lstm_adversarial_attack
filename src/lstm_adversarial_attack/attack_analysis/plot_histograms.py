import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack_analysis.perts_histogram_plotter as php


def main():



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--single_hist",
        "-s",
        nargs="*",
        action="append"
    )
    args_namespace = parser.parse_args()
    print(args_namespace.single_hist)




