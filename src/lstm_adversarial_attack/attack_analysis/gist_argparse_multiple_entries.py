import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--individual_histogram",
        action="append",
        nargs="+"
    )

    args_namespace = parser.parse_args()
    print(args_namespace.individual_histogram)