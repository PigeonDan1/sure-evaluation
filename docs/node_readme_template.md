# Node README Template

Use this structure for every versioned node under
`src/sure_eval/evaluation/nodes/<stage>/<name>/README.md`.

## Required Sections

```markdown
# <Human Title>

## Purpose

One paragraph explaining what the node does and what it does not do.

## Task Scenarios

- `<task>/<language>/<metric>` route or pipeline family that selects this node.
- Whether it is a default route, an alternative route, or an optional backend.

## Input

- Schema name from `manifest.yaml`.
- Required roles or fields.
- Alignment key and row format, when relevant.

## Output

- Output schema from `manifest.yaml`.
- Report fields or trace fields produced by this node.
- Whether higher or lower scores are better for scoring nodes.

## Versioned Computation

- Node id and version.
- Internal stages.
- Important algorithm, tokenizer, normalization, frontend, or aggregation rules.

## Runtime and Assets

- Runtime type: `in_process`, `pip`, `uv`, or `binary`.
- Optional packages, model checkpoints, binary env vars, and setup commands.
- Whether the node can run in the base package.

## Source and References

- Official upstream repository, model card, paper, or local implementation note.
- If no public source is known, say `External source not identified; this is a local compatibility implementation.`

## Limitations

- Known unsupported formats, languages, edge cases, or assumptions.
```

Keep repository paths relative. Do not document absolute local paths.
