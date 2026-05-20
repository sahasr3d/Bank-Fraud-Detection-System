Security and Data Handling

- Do not commit sensitive data, credentials, or API keys into the repository.
- Store raw datasets in a secure location and reference them via environment
  variables or configuration not stored in source control.
- If you discover a secret accidentally committed, rotate the secret immediately
  and remove it from history (e.g., `git filter-repo` or `git filter-branch`).
- Report security issues to the project owner.
