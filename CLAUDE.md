# Claude Code Instructions

## Project
Jekyll blog hosted on GitHub Pages at https://asanyaga.com (repo: asanyaga/asanyaga.github.io).

## Git workflow
- Claude Code works on a separate branch (worktree) each session — never commit directly to `main`.
- When a task is complete and the user confirms they are satisfied, do the following in order:
  1. Commit all changes with a clear commit message.
  2. Push the branch to GitHub (`git push -u origin <branch>`).
  3. Remind the user to open a PR at the GitHub URL printed after pushing, or offer to open it via `gh pr create` if the `gh` CLI is available.
  4. After the user confirms the PR is merged, remind them to:
     - Delete the remote branch (button on the merged PR page on GitHub).
     - Run `git worktree remove .claude/worktrees/<branch-name>` and `git branch -d <branch-name>` from `C:\Repos\asanyaga.github.io` to clean up locally.
