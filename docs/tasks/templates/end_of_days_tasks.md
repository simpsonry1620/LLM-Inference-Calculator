# End-of-Day Tasks Checklist Template for AI Assistant

***Instructions for AI Assistant:**

*   **Copy Template:** Before starting the end-of-day process, copy this file (`docs/tasks/templates/end_of_days_tasks.md`) to the `docs/tasks/in_progress/` directory.
*   **Rename File:** Append the current date to the new file name in `YYYY-MM-DD` format (e.g., `end_of_days_tasks_2023-10-27.md`).
*   **Perform Checklist:** Use the newly created file in the `in_progress` directory to track the completion of the checklist items below. Mark each item as completed.
*   **Confirm Completion:** Confirm each step is completed or explicitly state if a step is skipped and why.
*   **Archive:** Once all applicable steps are completed and confirmed, move the dated checklist file from `docs/tasks/in_progress/` to `docs/tasks/archive/`.

***

# End-of-Day Tasks Checklist for AI Assistant

Before committing and pushing changes to the repository, please perform the following tasks systematically:

## 1. Code Quality Checks

-   [ ] **Linting:** Run the project's linter (e.g., `pylint`, `flake8`, `ruff check .`) across all modified files and directories. Fix any reported errors or warnings.
-   [ ] **Formatting:** Ensure code formatting adheres to project standards (e.g., run `black .`, `ruff format .`).

## 2. Testing

-   [ ] **Run Unit Tests:** Execute the full suite of unit tests (e.g., `pytest`). Ensure all tests pass. Investigate and fix any failures.
-   [ ] **Run Integration Tests (if applicable):** Execute any integration tests to verify interactions between components.

## 3. Documentation Review

-   [ ] **Update Docstrings:** Review and update docstrings for any new or modified functions, classes, or modules.
-   [ ] **Update docs/*.md files:** Check if the main `README.md` a other documentation files (`docs/`) need updates to reflect the changes made (e.g., new features, usage instructions, dependency changes).
-   [ ] **Update README.md :** Check if the main `README.md` reflect the changes made (e.g., new features, usage instructions, dependency changes).
-   [ ] **Check `CHANGELOG.md` (if applicable):** Add an entry summarizing the changes for the next release.

## 4. Git Operations

-   [ ] **Review Changes:** Use `git status` and `git diff --staged` to review the changes one last time. Ensure only intended changes are staged.
-   [ ] **Craft Commit Message:** Write a clear, concise, and informative commit message following project conventions (e.g., Conventional Commits). The message should summarize *what* was changed and *why*.
-   [ ] **Commit:** Perform the `git commit`.
-   [ ] **Pull Latest Changes:** Run `git pull origin <your-branch>` to fetch and merge the latest changes from the remote repository, resolving any conflicts. Re-run tests if necessary after merging.
-   [ ] **Push:** Push the changes to the remote repository using `git push origin <your-branch>`.

## 5. Final Verification (Optional)

-   [ ] **Check CI/CD Pipeline:** If applicable, monitor the continuous integration pipeline to ensure the build and tests pass successfully after pushing.

---

*Instructions for the AI:* Please confirm each step is completed or explicitly state if a step is skipped and why.* 