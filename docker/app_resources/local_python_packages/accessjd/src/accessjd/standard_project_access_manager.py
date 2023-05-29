from pathlib import Path
import accessjd.jetbrains_access_manager as jam
import accessjd.simple_access_manager as sam


STANDARD_PROJECT_ROOT = Path("/home/devspace/project")


def get_standard_jetbrains_access_manager() -> sam.SimpleAccessManager:
    return jam.JetbrainsAccessManagerBuilder.from_project_root(
        project_root=STANDARD_PROJECT_ROOT).build()