import pprint
import accessjd.standard_project_access_manager as spam


def main():
    jb_access_manager = spam.get_standard_jetbrains_access_manager()
    condition_check_result = jb_access_manager.check_conditions()
    print(f"Safe to grant access: {condition_check_result.all_safe}")
    print("Condition error messages:")
    pprint.pprint(condition_check_result.bad_condition_messages)


if __name__ == "__main__":
    main()
