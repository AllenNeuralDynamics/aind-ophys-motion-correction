import version_info



current_branch, current_commit = version_info.get_current_branch_and_commit()
pipeline_version, url = version_info.get_pipeline_version()
print(f"CURRENT BRANCH {current_branch}")
print(f"CURRENT COMMIT {current_commit}")