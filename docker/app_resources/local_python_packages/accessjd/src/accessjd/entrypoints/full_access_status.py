import pprint
import accessjd.standard_project_access_manager as spam



def main():
    jb_access_manager = spam.get_standard_jetbrains_access_manager()
    print("Full accessjd status:")
    pprint.pp(jb_access_manager.full_root_access_dict("w"))


if __name__ == "__main__":
    main()