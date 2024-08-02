import inspect
import subprocess
import sys
from pathlib import Path
from typing import List


def redirect_output(log_file: Path, redirect_flags: List[str]):
    """
    Redirects the stdout and stderr of the calling script to the specified log file.

    :param log_file: The path to the log file where output should be redirected.
    :param redirect_flags: A list of command-line flags that trigger redirection.
    These flags will be removed from the arguments passed to the subprocess.
    :return: None
    """
    # Determine the calling script's file path
    caller_frame = inspect.stack()[1]
    script_name = caller_frame.filename

    print(f"Redirecting stdout and stderr to {str(log_file)}")

    log_file.parent.mkdir(exist_ok=True, parents=True)
    with log_file.open(mode="a") as output_file:
        # Filter out the redirection arg from original script command
        new_argv = [arg for arg in sys.argv if (arg not in redirect_flags )]

        # Call original script without its redirection flag
        result = subprocess.run(
            [sys.executable, "-u", script_name] + new_argv[1:],
            stdout=output_file,
            stderr=subprocess.STDOUT
        )
        sys.exit(result.returncode)
