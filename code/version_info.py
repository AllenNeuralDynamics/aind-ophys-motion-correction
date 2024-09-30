import os
import subprocess
from typing import Tuple

import codeocean


def get_pipeline_version() -> Tuple[str, str]:
    """Get version of code environment

    Returns
    -------
    str
        The version of the code environment.
    str
        The URL of the code environment.
    """
    domain = os.getenv("API_KEY_2")
    token = os.getenv("API_SECRET_2")
    if domain and token:
        client = codeocean.CodeOcean(domain=domain, token=token)
        capsule = client.capsules.get_capsule(capsule_id=os.getenv("CO_CAPSULE_ID"))
        return capsule.version, capsule.cloned_from_url
    else:
        return None, None


def get_current_branch_and_commit() -> Tuple[str, str]:
    """Get the current branch and commit hash of the git repository.

    Returns
    -------
    str
        The current branch.
    str
        The commit hash of the current branch.
    """
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return branch, commit
    except subprocess.CalledProcessError:
        return "Error: Not a git repository or git command failed", None


# Example usage
current_branch, current_commit = get_current_branch_and_commit()
