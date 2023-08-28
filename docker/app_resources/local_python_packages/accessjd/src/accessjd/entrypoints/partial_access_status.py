import pprint
import accessjd.standard_project_access_manager as spam


def main():
    jb_access_manager = spam.get_standard_jetbrains_access_manager()
    print("Partial accessjd status:")
    pprint.pprint(jb_access_manager.any_root_access_dict("w"))


if __name__ == "__main__":
    main()
