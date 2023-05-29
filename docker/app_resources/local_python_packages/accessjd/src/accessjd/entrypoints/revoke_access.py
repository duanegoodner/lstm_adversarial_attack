import accessjd.standard_project_access_manager as spam


def main():
    jb_access_manager = spam.get_standard_jetbrains_access_manager()
    jb_access_manager.revoke_access()


if __name__ == "__main__":
    main()