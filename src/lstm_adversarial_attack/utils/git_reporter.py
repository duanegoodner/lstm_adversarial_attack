import git
from pathlib import Path


class GitReporter:
    def __init__(self, output_root: Path):
        self.output_root = output_root
        assert output_root.is_dir()
        self.report_dir = self.create_report_dir()

        self.repo_root = Path(__file__).parent.parent.parent.parent
        self.repo = git.Repo(str(self.repo_root))

    def create_report_dir(self) -> Path:
        report_counter = 1
        report_dir_name = f"git_diff_report-{report_counter:02d}"
        report_dir_path = self.output_root / report_dir_name

        while report_dir_path.exists():
            report_counter += 1
            report_dir_name = f"git_diff_report-{report_counter:02d}"
            report_dir_path = self.output_root / report_dir_name

        report_dir_path.mkdir()
        return report_dir_path

    @property
    def current_branch_name(self) -> str | None:
        try:
            current_branch_name = self.repo.active_branch.name
        except TypeError:
            current_branch_name = None
            print(f"Detached HEAD at commit: {self.latest_commit_hash}")

        return current_branch_name

    @property
    def latest_commit_hash(self) -> str:
        return self.repo.head.commit.hexsha

    def save_branch_name(self):
        with (self.report_dir / "branch_name.txt").open(mode="w") as f:
            if self.current_branch_name:
                f.write(self.current_branch_name)
            else:
                f.write("None")

    def save_commit_hash(self):
        with (self.report_dir / "commit_hash.txt").open(mode="w") as f:
            f.write(self.latest_commit_hash)

    def save_git_diffs(self):
        changed_files = [item.a_path for item in self.repo.index.diff(None)]
        for file in changed_files:
            diff = self.repo.git.diff(file)
            if diff:
                diff_file = self.report_dir / file.replace("/", "_")
                with diff_file.open(mode="w") as f:
                    f.write(diff)




if __name__ == "__main__":
    git_reporter = GitReporter(output_root=Path(__file__).parent)
    git_reporter.save_branch_name()
    git_reporter.save_commit_hash()
    git_reporter.save_git_diffs()