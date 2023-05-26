import standard_project_access_manager as spam


def main():
    jb_access_manager = spam.get_standard_jetbrains_access_manager()
    print(jb_access_manager.is_safe_to_grant_access())


if __name__ == "__main__":
    main()
