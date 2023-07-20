import os
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


def most_recently_modified_file_named(
    target_filename: str, root_dir: Path
) -> Path:
    most_recent_file = None
    most_recent_time = 0

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == target_filename:
                file_path = os.path.join(dirpath, filename)
                file_time = os.path.getmtime(file_path)

                if file_time > most_recent_time:
                    most_recent_time = file_time
                    most_recent_file = file_path

    return Path(most_recent_file)
