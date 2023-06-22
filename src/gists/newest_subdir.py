from pathlib import Path


def get_newest_sub_dir(path: Path) -> Path | None:
    assert path.is_dir()
    sub_dirs = [item for item in path.iterdir() if item.is_dir()]
    if len(sub_dirs) != 0:
        return sorted(sub_dirs, key=lambda x: x.name)[-1]
    else:
        return None


print(
    get_newest_sub_dir(
        path=Path(__file__).parent.parent.parent / "data" / "model_assessments" / "single_fold_training"
    )
)
