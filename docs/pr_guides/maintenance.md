# Maintenance PR Guide

Use this for bug fixes, docs, tests, packaging, CI, or refactors that do not
change evaluation route semantics.

## Required Changes

- Keep edits scoped to the bug or maintenance target.
- Do not change `pipeline_id`, default routes, metric definitions, or node
  versions unless the PR is reclassified.
- Update docs when behavior or public commands change.
- Add regression tests for bug fixes.

## Required Tests

- Focused regression tests for the affected behavior.
- `ruff check` when Python code changes.
- `git diff --check` for all PRs.
- Full or broader tests when shared dispatch, CLI, contracts, or identity code
  changes.

## PR Must State

- Why no evaluation identity changed.
- Tests run.
- User-visible behavior, if any.
