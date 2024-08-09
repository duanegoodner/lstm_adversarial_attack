import os
import subprocess
from pathlib import Path

def create_directory(directory_name="git_diff_report"):
    # Create a directory to store the results
    Path(directory_name).mkdir(parents=True, exist_ok=True)
    return directory_name

def get_latest_commit_hash():
    try:
        # Capture the output of the git command
        result = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        latest_commit_hash = result.stdout.decode('utf-8').strip()
        return latest_commit_hash
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while retrieving latest commit hash: {e.stderr.decode('utf-8')}")
        return None

def save_commit_hash_to_file(commit_hash, directory):
    # Save the commit hash to a text file
    with open(os.path.join(directory, "latest_commit_hash.txt"), "w") as f:
        f.write(commit_hash)

def generate_git_diff(directory):
    # Get the list of files that differ from the latest commit
    result = subprocess.run(["git", "diff", "--name-only"], stdout=subprocess.PIPE)
    changed_files = result.stdout.decode('utf-8').splitlines()

    # For each changed file, generate a git diff and save it to a file
    for file in changed_files:
        diff_result = subprocess.run(["git", "diff", file], stdout=subprocess.PIPE)
        diff_output = diff_result.stdout.decode('utf-8')

        if diff_output:
            diff_file_name = os.path.join(directory, f"{Path(file).stem}_diff.txt")
            with open(diff_file_name, "w") as diff_file:
                diff_file.write(diff_output)

def main():
    directory = create_directory()

    # Step 1: Get the latest commit hash and save it to a file
    latest_commit_hash = get_latest_commit_hash()
    save_commit_hash_to_file(latest_commit_hash, directory)

    # Step 2: Generate git diff files for any files that have local changes
    generate_git_diff(directory)

if __name__ == "__main__":
    main()
