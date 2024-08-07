import os
from enum import Enum, auto
from pathlib import Path


def latest_content_modification_time(directory: Path):
    most_recent_time = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            modification_time = os.path.getmtime(path)
            if modification_time > most_recent_time:
                most_recent_time = modification_time

    return most_recent_time


def most_recently_modified_subdir(root_path: Path) -> Path:
    assert root_path.is_dir()
    sub_dirs = [item for item in root_path.iterdir() if item.is_dir()]
    assert len(sub_dirs) > 0
    sorted_sub_dirs = sorted(
        sub_dirs, key=lambda x: latest_content_modification_time(x)
    )
    return sorted_sub_dirs[-1]


class StringComparisonType(Enum):
    EXACT_MATCH = auto()
    SUFFIX = auto()
    PREFIX = auto()
    CONTAINS = auto()


def compare_strings(
    required_str: str, target_str: str, comparison_type: StringComparisonType
):
    comparison_dispatch = {
        StringComparisonType.EXACT_MATCH: target_str.__eq__,
        StringComparisonType.SUFFIX: target_str.endswith,
        StringComparisonType.PREFIX: target_str.startswith,
        StringComparisonType.CONTAINS: target_str.__contains__,
    }

    return comparison_dispatch[comparison_type](required_str)


def latest_modified_file_with_name_condition(
    component_string: str,
    root_dir: Path,
    comparison_type: StringComparisonType = StringComparisonType.EXACT_MATCH,
) -> Path:
    most_recent_file = None
    most_recent_time = 0

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if compare_strings(
                required_str=component_string,
                target_str=filename,
                comparison_type=comparison_type,
            ):
                # if filename == component_string:
                file_path = os.path.join(dirpath, filename)
                file_time = os.path.getmtime(file_path)

                if file_time > most_recent_time:
                    most_recent_time = file_time
                    most_recent_file = file_path

    return Path(most_recent_file)


def get_latest_sequential_child_dir(root_dir: Path) -> Path:
    child_dir_paths = [path for path in root_dir.iterdir() if path.is_dir()]
    return max(child_dir_paths)

def get_latest_sequential_child_dirname(root_dir: Path) -> str:
    latest_dir = get_latest_sequential_child_dir(root_dir=root_dir)
    return latest_dir.name


