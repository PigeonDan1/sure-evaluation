# Recommendations

This file is superseded by [PLAN.md](./PLAN.md).

The current release policy is:

- Keep local virtual environments, checkpoints, runtime manifests, and logs on disk for development.
- Keep those runtime assets out of git through `.gitignore`.
- Make the base package installable without heavyweight model or toolkit dependencies.
- Prepare heavyweight metric/model nodes through explicit, optional `sure-eval env ...` commands.

