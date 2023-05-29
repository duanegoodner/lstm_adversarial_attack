from pathlib import Path
import accessjd.jetbrains_access_manager as jam


def main():
    """
    For functionality check and troubleshooting
    :return: 3-member tuple consisting of JetBrainsAccess Maager object,
    bool indicating if jetbrains user has any write accessjd, and a bool
    indicating if it is safe to grant accessjd to jetbrains user
    """
    project_root = Path(__file__).parent.parent
    access_mgr = jam.JetbrainsAccessManagerBuilder.from_project_root(
        project_root=project_root
    ).build()
    has_any_write_access = access_mgr.has_any_access("w")
    is_safe_to_grant_access = access_mgr.is_safe_to_grant_access()

    return access_mgr, has_any_write_access, is_safe_to_grant_access


if __name__ == "__main__":
    mgr, any_write_access, safe = main()
