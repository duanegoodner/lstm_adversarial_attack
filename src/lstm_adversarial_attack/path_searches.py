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


def subdir_with_latest_content_modification(root_path: Path) -> Path:
    assert root_path.is_dir()
    sub_dirs = [item for item in root_path.iterdir() if item.is_dir()]
    assert len(sub_dirs) > 0
    sorted_sub_dirs = sorted(
        sub_dirs, key=lambda x: latest_content_modification_time(x)
    )
    return sorted_sub_dirs[-1]
