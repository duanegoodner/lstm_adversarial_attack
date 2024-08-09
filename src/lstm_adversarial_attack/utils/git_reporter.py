import os
from pathlib import Path
import git

def create_directory(directory_name="git_diff_report"):
    # Create a directory to store the results
    Path(directory_name).mkdir(parents=True, exist_ok=True)
    return directory_name

def get_latest_commit_hash(repo):
    # Get the latest commit hash
    latest_commit = repo.head.commit
    return latest_commit.hexsha

def save_commit_hash_to_file(commit_hash, directory):
    # Save the commit hash to a text file
    with open(os.path.join(directory, "latest_commit_hash.txt"), "w") as f:
        f.write(commit_hash)

def generate_git_diff(repo, directory):
    # Check for changed files in the working directory compared to the latest commit
    changed_files = [item.a_path for item in repo.index.diff(None)]

    # For each changed file, generate a git diff and save it to a file
    for file in changed_files:
        diff = repo.git.diff(file)
        if diff:
            diff_file_name = os.path.join(directory, f"{Path(file).stem}_diff.txt")
            with open(diff_file_name, "w") as diff_file:
                diff_file.write(diff)

def main():
    # Initialize the repo object from the current working directory
    repo = git.Repo(str(Path(__file__).parent.parent.parent.parent))

    # Ensure the repository is marked as safe
    repo.config_writer().set_value("safe", "directory", os.getcwd()).release()

    directory = create_directory()

    # Step 1: Get the latest commit hash and save it to a file
    latest_commit_hash = get_latest_commit_hash(repo)
    save_commit_hash_to_file(latest_commit_hash, directory)

    # Step 2: Generate git diff files for any files that have local changes
    generate_git_diff(repo, directory)

if __name__ == "__main__":
    main()
