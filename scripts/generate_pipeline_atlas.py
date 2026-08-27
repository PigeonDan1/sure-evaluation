#!/usr/bin/env python3
"""Generate the pipeline atlas from the committed catalog.

Reads docs/pipeline_catalog.jsonl (regenerate it first via
scripts/generate_pipeline_catalog.py after route changes) and writes:

- docs/atlas/pipeline_atlas.svg        animated map for the README (light)
- docs/atlas/pipeline_atlas_dark.svg   animated map for the README (dark)
- docs/atlas/index.html                interactive map: search, filter, hover tracing

The script only reads the committed catalog, so it needs no installed
package or node environments.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG = REPO_ROOT / "docs" / "pipeline_catalog.jsonl"
OUT_DIR = REPO_ROOT / "docs" / "atlas"

STAGES = ("frontend", "transcription", "validation", "normalization", "scoring")

# Color families: the eight validated categorical hues go to the eight largest
# task groups; remaining small task groups share a neutral gray. Identity is
# always carried by labels, never by color alone.
FAMILY_ORDER = ("asr", "tts", "vc", "tse", "s2tt", "se", "kws", "vad")
PALETTE = {
    "light": {
        "asr": "#2a78d6", "tts": "#eb6834", "vc": "#1baf7a", "tse": "#eda100",
        "s2tt": "#e87ba4", "se": "#008300", "kws": "#4a3aa7", "vad": "#e34948",
        "other": "#8a8a86",
        "surface": "#fcfcfb", "lane": "#64748b0d", "border": "#e3e1dc",
        "ink": "#0b0b0b", "ink2": "#52514e", "ink3": "#8a8a86",
    },
    "dark": {
        "asr": "#3987e5", "tts": "#d95926", "vc": "#199e70", "tse": "#c98500",
        "s2tt": "#d55181", "se": "#008300", "kws": "#9085e9", "vad": "#e66767",
        "other": "#8f8e88",
        "surface": "#1a1a19", "lane": "#94a3b80f", "border": "#3a3936",
        "ink": "#ffffff", "ink2": "#c3c2b7", "ink3": "#8f8e88",
    },
}

NODE_W, NODE_H, NODE_GAP = 172, 26, 9
TASK_W, TASK_H, TASK_GAP = 148, 32, 8
WIDTH = 1560
CYCLE_SLOT_S = 1.7
DRAW_STAGGER_MS = 12
DRAW_LEAD_S = 2.2

SANS = "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif"
MONO = "ui-monospace, 'SF Mono', Menlo, Consolas, monospace"


def family_of(task: str) -> str:
    return task.lower() if task.lower() in FAMILY_ORDER else "other"


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def load_catalog() -> list[dict[str, Any]]:
    rows = []
    for line in CATALOG.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_layout(rows: list[dict[str, Any]]) -> dict[str, Any]:
    atomic = [r for r in rows if r["pipeline_kind"] == "atomic"]
    bundles = [r for r in rows if r["pipeline_kind"] != "atomic"]
    tasks = sorted(
        {r["task"] for r in rows},
        key=lambda t: (-len([r for r in atomic if r["task"] == t]), t),
    )
    task_idx = {t: i for i, t in enumerate(tasks)}

    # One drawn thread per scoring node so multi-metric pipelines fan out in
    # the scoring lane instead of collapsing onto one ribbon.
    threads = []
    for row in atomic:
        by_stage: dict[str, list[str]] = {}
        for node in row["nodes"]:
            by_stage.setdefault(node.split("/")[0], []).append(node)
        stops = [by_stage[s][0] for s in STAGES[:-1] if s in by_stage]
        for scoring in by_stage.get("scoring", [None]):
            threads.append({"row": row, "stops": stops + ([scoring] if scoring else [])})

    # Threads with an identical stop sequence (same chain, different language)
    # would overlap pixel-perfectly; fan them into parallel strands instead.
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for thread in threads:
        groups.setdefault((thread["row"]["task"], *thread["stops"]), []).append(thread)
    for group in groups.values():
        for i, thread in enumerate(group):
            offset = (i - (len(group) - 1) / 2) * 2.4
            thread["dy"] = max(-10.0, min(10.0, offset))

    node_use: dict[str, dict[str, Any]] = {}
    for thread in threads:
        for node in thread["stops"]:
            use = node_use.setdefault(node, {"count": 0, "tasks": {}})
            use["count"] += 1
            task = thread["row"]["task"]
            use["tasks"][task] = use["tasks"].get(task, 0) + 1

    def dominant_task_idx(node: str) -> int:
        return min(task_idx[t] for t in node_use[node]["tasks"])

    stage_nodes = {
        stage: sorted(
            (n for n in node_use if n.startswith(stage + "/")),
            key=lambda n: (dominant_task_idx(n), -node_use[n]["count"], n),
        )
        for stage in STAGES
    }

    max_stack = max(len(v) for v in stage_nodes.values())
    height = max(len(tasks) * (TASK_H + TASK_GAP), max_stack * (NODE_H + NODE_GAP)) + 110

    def lane_x(i: int) -> float:
        return 96 + i * (WIDTH - 260) / (len(STAGES) + 1)

    def center_ys(n: int, item_h: float, gap: float) -> list[float]:
        total = n * item_h + (n - 1) * gap
        y0 = 60 + (height - 70 - total) / 2
        return [y0 + i * (item_h + gap) + item_h / 2 for i in range(n)]

    task_y = dict(zip(tasks, center_ys(len(tasks), TASK_H, TASK_GAP)))
    node_pos = {}
    for si, stage in enumerate(STAGES):
        ys = center_ys(len(stage_nodes[stage]), NODE_H, NODE_GAP)
        for node, y in zip(stage_nodes[stage], ys):
            node_pos[node] = (lane_x(si + 1), y)

    return {
        "atomic": atomic, "bundles": bundles, "tasks": tasks, "task_idx": task_idx,
        "threads": threads, "node_use": node_use, "node_pos": node_pos,
        "task_y": task_y, "lane_x": lane_x, "height": height,
        "report_x": lane_x(len(STAGES) + 1), "report_y": height / 2,
    }


def ribbon_path(layout: dict[str, Any], thread: dict[str, Any]) -> str:
    dy = thread.get("dy", 0.0)
    points = [(layout["lane_x"](0) + TASK_W / 2, layout["task_y"][thread["row"]["task"]] + dy)]
    for node in thread["stops"]:
        x, y = layout["node_pos"][node]
        points.extend([(x - NODE_W / 2, y + dy), (x + NODE_W / 2, y + dy)])
    points.append((layout["report_x"] - 60, layout["report_y"] + dy))
    parts = [f"M {points[0][0]:.1f} {points[0][1]:.1f}"]
    for i in range(1, len(points)):
        (x1, y1), (x2, y2) = points[i - 1], points[i]
        if y1 == y2 and i % 2 == 0:
            parts.append(f"L {x2:.1f} {y2:.1f}")
        else:
            mx = (x1 + x2) / 2
            parts.append(f"C {mx:.1f} {y1:.1f}, {mx:.1f} {y2:.1f}, {x2:.1f} {y2:.1f}")
    return " ".join(parts)


def readme_svg_style(colors: dict[str, str], n_tasks: int) -> str:
    cycle = DRAW_LEAD_S + n_tasks * CYCLE_SLOT_S
    slot_pct = CYCLE_SLOT_S / cycle * 100
    return f"""
  text {{ font-family: {SANS}; fill: {colors["ink"]}; }}
  .mono {{ font-family: {MONO}; }}
  .lane {{ fill: {colors["lane"]}; stroke: {colors["border"]}; }}
  .lane-label {{ fill: {colors["ink3"]}; font-size: 11px; letter-spacing: .18em; font-weight: 600; }}
  .cnt {{ fill: {colors["ink3"]}; font-size: 10px; }}
  .box {{ fill: {colors["surface"]}; stroke: {colors["border"]}; }}
  .blk text {{ font-size: 12px; }}
  .task-label {{ font-size: 13px; font-weight: 600; }}
  .r {{
    fill: none; stroke-linecap: round; stroke-width: 3.2; opacity: .3;
    stroke-dasharray: 1; stroke-dashoffset: 1;
    animation: draw .9s cubic-bezier(.16, 1, .3, 1) forwards,
               cyc {cycle:.1f}s linear infinite;
    animation-delay: var(--d), var(--cd);
  }}
  @keyframes draw {{
    0% {{ stroke-dashoffset: 1; opacity: .7; }}
    100% {{ stroke-dashoffset: 0; opacity: .3; }}
  }}
  @keyframes cyc {{
    0% {{ opacity: .3; stroke-width: 3.2; }}
    {slot_pct * 0.12:.2f}% {{ opacity: .95; stroke-width: 5; }}
    {slot_pct * 0.88:.2f}% {{ opacity: .95; stroke-width: 5; }}
    {slot_pct:.2f}% {{ opacity: .3; stroke-width: 3.2; }}
    100% {{ opacity: .3; stroke-width: 3.2; }}
  }}
  .blk--task {{ opacity: .8; animation: tcyc {cycle:.1f}s linear infinite; animation-delay: var(--cd); }}
  @keyframes tcyc {{
    0% {{ opacity: .8; }}
    {slot_pct * 0.12:.2f}% {{ opacity: 1; }}
    {slot_pct * 0.88:.2f}% {{ opacity: 1; }}
    {slot_pct:.2f}% {{ opacity: .8; }}
    100% {{ opacity: .8; }}
  }}
  @media (prefers-reduced-motion: reduce) {{
    .r, .blk--task {{ animation: none; stroke-dashoffset: 0; opacity: .3; }}
    .blk--task {{ opacity: .9; }}
  }}
"""


def svg_markup(layout: dict[str, Any], theme: str, mode: str) -> str:
    colors = PALETTE[theme]
    height = layout["height"]

    def color(fam: str) -> str:
        return f"var(--task-{fam})" if mode == "app" else colors[fam]

    parts = []
    for si, stage in enumerate(STAGES):
        x = layout["lane_x"](si + 1)
        parts.append(
            f'<rect class="lane" x="{x - NODE_W / 2 - 14:.1f}" y="30" '
            f'width="{NODE_W + 28}" height="{height - 50:.1f}" rx="10"/>'
            f'<text class="lane-label" x="{x - NODE_W / 2 - 4:.1f}" y="52">{stage.upper()}</text>'
        )
    parts.append(
        f'<text class="lane-label" x="{layout["report_x"] - 60:.1f}" y="52">REPORT</text>'
    )

    node_threads: dict[str, list[int]] = {}
    for i, thread in enumerate(layout["threads"]):
        for node in thread["stops"]:
            node_threads.setdefault(node, []).append(i)

    for i, thread in enumerate(layout["threads"]):
        row = thread["row"]
        fam = family_of(row["task"])
        cycle_delay = DRAW_LEAD_S + layout["task_idx"][row["task"]] * CYCLE_SLOT_S
        style = f"--d:{i * DRAW_STAGGER_MS}ms;--cd:{cycle_delay:.1f}s"
        chain = " → ".join(n.split("/")[1] for n in row["nodes"]) + " → report"
        key = esc(f'{row["pipeline_id"]} {" ".join(row["nodes"])} '
                  f'{row.get("language", "")} {row["metric"]} {row["task"]}'.lower())
        tip = esc(
            f'<b>{esc(row["task"])}</b> <span class="t-tag">{esc(row.get("language", "n/a"))}'
            f' · {esc(row["metric"])}</span><br><span class="t-id">{esc(row["pipeline_id"])}'
            f'</span><br><span class="t-tag">{esc(chain)}</span>'
        )
        data = (f' data-i="{i}" data-task="{esc(row["task"])}" data-key="{key}" data-tip="{tip}"'
                if mode == "app" else "")
        parts.append(
            f'<path class="r" pathLength="1" style="{style}" stroke="{color(fam)}"'
            f'{data} d="{ribbon_path(layout, thread)}"/>'
        )

    for task in layout["tasks"]:
        fam = family_of(task)
        y = layout["task_y"][task]
        x0 = layout["lane_x"](0) - TASK_W / 2
        count = len([r for r in layout["atomic"] if r["task"] == task])
        cycle_delay = DRAW_LEAD_S + layout["task_idx"][task] * CYCLE_SLOT_S
        data = f' data-task="{esc(task)}"' if mode == "app" else ""
        parts.append(
            f'<g class="blk blk--task" style="--cd:{cycle_delay:.1f}s;--c:{color(fam)}"{data}>'
            f'<rect class="box" x="{x0:.1f}" y="{y - TASK_H / 2:.1f}" width="{TASK_W}" height="{TASK_H}" rx="7"/>'
            f'<rect x="{x0:.1f}" y="{y - TASK_H / 2:.1f}" width="4" height="{TASK_H}" rx="2" fill="{color(fam)}"/>'
            f'<text class="task-label" x="{x0 + 14:.1f}" y="{y + 4:.1f}">{esc(task)}</text>'
            f'<text class="cnt" x="{x0 + TASK_W - 12:.1f}" y="{y + 4:.1f}" text-anchor="end">{count}</text></g>'
        )

    for node, (x, y) in layout["node_pos"].items():
        use = layout["node_use"][node]
        fam = family_of(layout["tasks"][min(layout["task_idx"][t] for t in use["tasks"])])
        name = node.split("/")[1]
        label = name if len(name) <= 18 else name[:17] + "…"
        if mode == "app":
            per = " · ".join(f"{t}×{c}" for t, c in use["tasks"].items())
            tip = esc(f'<b>{esc(node)}</b> <span class="t-tag">crossed by {use["count"]} '
                      f'pipelines</span><br><span class="t-tag">{esc(per)}</span>')
            data = (f' data-node="{esc(node)}" data-tip="{tip}"'
                    f' data-threads="{",".join(map(str, node_threads[node]))}"')
        else:
            data = ""
        parts.append(
            f'<g class="blk" style="--c:{color(fam)}"{data}>'
            f'<rect class="box" x="{x - NODE_W / 2:.1f}" y="{y - NODE_H / 2:.1f}"'
            f' width="{NODE_W}" height="{NODE_H}" rx="6"/>'
            f'<text class="mono" x="{x - NODE_W / 2 + 10:.1f}" y="{y + 4:.1f}">{esc(label)}</text>'
            f'<text class="cnt" x="{x + NODE_W / 2 - 8:.1f}" y="{y + 4:.1f}" text-anchor="end">'
            f'×{use["count"]}</text></g>'
        )

    rx, ry = layout["report_x"], layout["report_y"]
    data = ' data-report="1"' if mode == "app" else ""
    parts.append(
        f'<g class="blk"{data}><rect class="box" x="{rx - 60:.1f}" y="{ry - 26:.1f}"'
        f' width="150" height="52" rx="9"/>'
        f'<text class="task-label" x="{rx - 44:.1f}" y="{ry - 4:.1f}">{len(layout["atomic"])} pipelines</text>'
        f'<text class="cnt mono" x="{rx - 44:.1f}" y="{ry + 14:.1f}">report.json</text></g>'
    )

    if mode == "readme":
        style = readme_svg_style(PALETTE[theme], len(layout["tasks"]))
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {layout["height"]:.0f}"'
            f' font-family="{esc(SANS)}" role="img" aria-label="SURE pipeline atlas">'
            f"<title>SURE pipeline atlas: every atomic pipeline as a ribbon from task to report"
            f"</title><style>{style}</style>"
            f'<rect width="{WIDTH}" height="{layout["height"]:.0f}" fill="{PALETTE[theme]["surface"]}"/>'
            + "".join(parts) + "</svg>"
        )
    return (
        f'<svg class="atlas-svg" viewBox="0 0 {WIDTH} {layout["height"]:.0f}" role="img"'
        f' aria-label="SURE pipeline atlas">' + "".join(parts) + "</svg>"
    )


APP_CSS = """
.atlas-root {
  color-scheme: light;
  --surface-1: #fcfcfb; --surface-2: #f4f3f1; --lane: rgba(100, 116, 139, .05);
  --border-1: #e3e1dc; --ink: #0b0b0b; --ink-2: #52514e; --ink-3: #8a8a86;
  --task-asr: #2a78d6; --task-tts: #eb6834; --task-vc: #1baf7a; --task-tse: #eda100;
  --task-s2tt: #e87ba4; --task-se: #008300; --task-kws: #4a3aa7; --task-vad: #e34948;
  --task-other: #8a8a86; --fast: 150ms;
  font-family: %(sans)s;
  background: var(--surface-1); color: var(--ink);
  padding: clamp(12px, 3vw, 32px); min-height: 100vh; box-sizing: border-box;
}
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) .atlas-root { %(dark)s } }
:root[data-theme="dark"] .atlas-root { %(dark)s }
.atlas-root header h1 { margin: 0 0 4px; font-size: clamp(1.15rem, 2.5vw, 1.5rem); }
.atlas-root header p { margin: 0; color: var(--ink-2); font-size: .85rem; }
.hud { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 16px 0 10px; }
.search { flex: 1 1 220px; max-width: 380px; padding: 7px 12px; border: 1px solid var(--border-1);
  border-radius: 999px; background: var(--surface-2); color: var(--ink); font-size: .82rem; outline: none; }
.search:focus { border-color: var(--task-asr); }
.stats { margin-left: auto; color: var(--ink-3); font-size: .78rem; font-variant-numeric: tabular-nums; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
.chip { display: inline-flex; align-items: center; gap: 6px; padding: 4px 11px;
  border: 1px solid var(--border-1); border-radius: 999px; background: var(--surface-1);
  color: var(--ink-2); font-size: .75rem; cursor: pointer;
  transition: border-color var(--fast), background var(--fast); }
.chip .dot { width: 8px; height: 8px; border-radius: 50%%; background: var(--c, var(--task-other)); }
.chip:hover { border-color: var(--c, var(--ink-3)); }
.chip.on { background: var(--surface-2); border-color: var(--c, var(--ink-3)); color: var(--ink); }
.frame { border: 1px solid var(--border-1); border-radius: 14px; background: var(--surface-1); overflow-x: auto; }
.atlas-svg { display: block; min-width: 1000px; width: 100%%; height: auto; }
.lane { fill: var(--lane); stroke: var(--border-1); }
.lane-label { fill: var(--ink-3); font-size: 11px; letter-spacing: .18em; font-weight: 600; }
.r { fill: none; stroke-width: 3.2; stroke-linecap: round; opacity: .3; cursor: pointer;
  transition: opacity var(--fast), stroke-width var(--fast); }
.atlas-svg.animate .r { stroke-dasharray: 1; stroke-dashoffset: 1;
  animation: draw .9s cubic-bezier(.16, 1, .3, 1) forwards; animation-delay: var(--d); }
@keyframes draw { to { stroke-dashoffset: 0; } }
.r.lit { opacity: .95; stroke-width: 5; }
.atlas-svg.animate .r.lit.flow { stroke-dasharray: .06 .014; stroke-dashoffset: 0;
  animation: flow .9s linear infinite; }
@keyframes flow { to { stroke-dashoffset: -.074; } }
.r.dim { opacity: .05; }
.r.off { opacity: .02; pointer-events: none; }
.blk { cursor: pointer; }
.blk .box { fill: var(--surface-1); stroke: var(--border-1); transition: stroke var(--fast); }
.blk text { fill: var(--ink); font-size: 12px; }
.blk .mono { font-family: %(mono)s; }
.blk .cnt { fill: var(--ink-3); font-size: 10px; }
.blk:hover .box, .blk.lit .box { stroke: var(--c, var(--ink-3)); stroke-width: 1.5; }
.blk.dim { opacity: .35; }
.task-label { font-weight: 600; font-size: 13px; }
.tip { position: fixed; z-index: 10; pointer-events: none; max-width: 380px; padding: 8px 12px;
  border: 1px solid var(--border-1); border-radius: 8px; background: var(--surface-1); color: var(--ink);
  font-size: .75rem; line-height: 1.5; box-shadow: 0 6px 24px rgba(0, 0, 0, .14);
  opacity: 0; transform: translateY(4px); transition: opacity var(--fast), transform var(--fast); }
.tip.show { opacity: 1; transform: translateY(0); }
.tip .t-id { font-family: %(mono)s; word-break: break-all; color: var(--ink-2); }
.tip .t-tag { color: var(--ink-3); }
.tbl { width: 100%%; border-collapse: collapse; font-size: .78rem; }
.tbl th, .tbl td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--border-1); }
.tbl th { color: var(--ink-3); font-weight: 600; font-size: .72rem; letter-spacing: .06em; }
.tbl td.mono { font-family: %(mono)s; word-break: break-all; color: var(--ink-2); }
.tbl .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%%;
  background: var(--c); margin-right: 6px; }
.hidden { display: none; }
footer.foot { margin-top: 10px; color: var(--ink-3); font-size: .72rem; }
@media (prefers-reduced-motion: reduce) {
  .atlas-svg.animate .r { animation: none; stroke-dasharray: none; stroke-dashoffset: 0; }
  .atlas-svg.animate .r.lit { animation: none; stroke-dasharray: none; }
}
"""

APP_DARK_TOKENS = """color-scheme: dark;
  --surface-1: #1a1a19; --surface-2: #232322; --lane: rgba(148, 163, 184, .06);
  --border-1: #3a3936; --ink: #ffffff; --ink-2: #c3c2b7; --ink-3: #8f8e88;
  --task-asr: #3987e5; --task-tts: #d95926; --task-vc: #199e70; --task-tse: #c98500;
  --task-s2tt: #d55181; --task-se: #008300; --task-kws: #9085e9; --task-vad: #e66767;
  --task-other: #8f8e88;"""

APP_JS = """
const svg = document.querySelector(".atlas-svg");
const ribbons = [...svg.querySelectorAll(".r")];
const blocks = [...svg.querySelectorAll(".blk")];
const tip = document.getElementById("tip");
const activeTasks = new Set();
let query = "";
let pinned = null;

const matches = (r) => (!activeTasks.size || activeTasks.has(r.dataset.task)) &&
  (!query || r.dataset.key.includes(query));

function applyFilter() {
  ribbons.forEach((r) => r.classList.toggle("off", !matches(r)));
  blocks.forEach((b) => {
    if (b.dataset.node) b.classList.toggle("dim",
      !b.dataset.threads.split(",").some((i) => !ribbons[+i].classList.contains("off")));
    if (b.dataset.task) b.classList.toggle("dim",
      !ribbons.some((r) => r.dataset.task === b.dataset.task && !r.classList.contains("off")));
  });
  const shown = new Set(ribbons.filter((r) => !r.classList.contains("off"))
    .map((r) => r.dataset.tip));
  document.getElementById("stats").textContent =
    `${shown.size} / ${TOTALS.atomic} pipelines · ${TOTALS.nodes} nodes`;
  document.querySelectorAll("#tblFrame tr[data-key]").forEach((tr) => {
    tr.classList.toggle("hidden", !((!activeTasks.size || activeTasks.has(tr.dataset.task)) &&
      (!query || tr.dataset.key.includes(query))));
  });
}

function light(litSet) {
  const flowing = !!litSet && litSet.size <= 3;
  ribbons.forEach((r, i) => {
    const on = litSet && litSet.has(i) && !r.classList.contains("off");
    r.classList.toggle("lit", !!on);
    r.classList.toggle("flow", !!on && flowing);
    r.classList.toggle("dim", litSet ? !on && !r.classList.contains("off") : false);
  });
  blocks.forEach((b) => {
    if (!b.dataset.node) return;
    b.classList.toggle("lit", !!litSet &&
      b.dataset.threads.split(",").some((i) => litSet.has(+i) && !ribbons[+i].classList.contains("off")));
  });
}
const taskSet = (task) => new Set(ribbons.filter((r) => r.dataset.task === task).map((r) => +r.dataset.i));

function showTip(html, e) {
  tip.innerHTML = html;
  tip.classList.add("show");
  tip.style.left = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 14) + "px";
  tip.style.top = Math.min(e.clientY + 14, innerHeight - tip.offsetHeight - 14) + "px";
}
const hideTip = () => tip.classList.remove("show");

svg.addEventListener("pointermove", (e) => {
  const rb = e.target.closest(".r");
  const blk = e.target.closest(".blk");
  if (rb) {
    if (!pinned) light(new Set([+rb.dataset.i]));
    showTip(rb.dataset.tip, e);
  } else if (blk && blk.dataset.node) {
    if (!pinned) light(new Set(blk.dataset.threads.split(",").map(Number)));
    showTip(blk.dataset.tip, e);
  } else if (blk && blk.dataset.task) {
    if (!pinned) light(taskSet(blk.dataset.task));
    hideTip();
  } else if (blk) {
    if (!pinned) light(new Set(ribbons.map((_, i) => i)));
    showTip(`<b>report.json</b><br><span class="t-tag">one report per run: score plus the full` +
      ` pipeline description, both written to disk${TOTALS.bundles ?
      `; ${TOTALS.bundles} bundle pipelines aggregate their atomic members` : ""}</span>`, e);
  } else if (!pinned) { light(null); hideTip(); }
});
svg.addEventListener("pointerleave", () => { if (!pinned) light(null); hideTip(); });
svg.addEventListener("click", (e) => {
  const rb = e.target.closest(".r");
  const blk = e.target.closest(".blk");
  if (rb) { pinned = pinned === rb ? null : rb; light(pinned ? new Set([+rb.dataset.i]) : null); }
  else if (blk && blk.dataset.task) {
    const chip = document.querySelector(`.chip[data-task="${blk.dataset.task}"]`);
    if (chip) chip.click();
  } else { pinned = null; light(null); }
});

const chipRow = document.getElementById("chips");
chipRow.addEventListener("click", (e) => {
  const c = e.target.closest(".chip");
  if (!c) return;
  if (c.dataset.all) activeTasks.clear();
  else c.classList.contains("on") ? activeTasks.delete(c.dataset.task) : activeTasks.add(c.dataset.task);
  chipRow.querySelectorAll(".chip").forEach((x) =>
    x.classList.toggle("on", x.dataset.all ? !activeTasks.size : activeTasks.has(x.dataset.task)));
  applyFilter();
});
document.getElementById("q").addEventListener("input", (e) => {
  query = e.target.value.trim().toLowerCase();
  applyFilter();
});
const frames = { viewFlow: "flowFrame", viewTable: "tblFrame" };
Object.keys(frames).forEach((btn) => document.getElementById(btn).addEventListener("click", () => {
  Object.entries(frames).forEach(([b, f]) => {
    document.getElementById(b).classList.toggle("on", b === btn);
    document.getElementById(f).classList.toggle("hidden", b !== btn);
  });
}));
applyFilter();
"""


def app_html(layout: dict[str, Any]) -> str:
    chips = ['<button class="chip on" data-all="1">All tasks</button>']
    for task in layout["tasks"]:
        chips.append(
            f'<button class="chip" data-task="{esc(task)}" style="--c:var(--task-{family_of(task)})">'
            f'<span class="dot"></span>{esc(task)}</button>'
        )
    rows = []
    for row in layout["atomic"] + layout["bundles"]:
        chain = " → ".join(n.split("/")[1] for n in row["nodes"])
        key = esc(f'{row["pipeline_id"]} {" ".join(row["nodes"])} '
                  f'{row.get("language", "")} {row["metric"]} {row["task"]}'.lower())
        mark = "" if row["pipeline_kind"] == "atomic" else " ⛓"
        rows.append(
            f'<tr data-task="{esc(row["task"])}" data-key="{key}">'
            f'<td><span class="dot" style="--c:var(--task-{family_of(row["task"])})"></span>'
            f'{esc(row["task"])}{mark}</td><td>{esc(row.get("language", "n/a"))}</td>'
            f'<td>{esc(row["metric"])}</td><td class="mono">{esc(row["pipeline_id"])}</td>'
            f'<td class="mono">{esc(chain)}</td></tr>'
        )
    totals = json.dumps({
        "atomic": len(layout["atomic"]), "bundles": len(layout["bundles"]),
        "nodes": len(layout["node_use"]),
    })
    css = APP_CSS % {"sans": SANS, "mono": MONO, "dark": APP_DARK_TOKENS}
    stage_flow = " / ".join(STAGES)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SURE Pipeline Atlas</title>
<style>{css}</style>
</head>
<body>
<div class="atlas-root">
  <header>
    <h1>SURE Pipeline Atlas</h1>
    <p>Each colored ribbon is one reviewable evaluation declaration: a task flows through
    {esc(stage_flow)} nodes and lands as one report.
    Generated from <code>docs/pipeline_catalog.jsonl</code> by
    <code>scripts/generate_pipeline_atlas.py</code>.</p>
  </header>
  <div class="hud">
    <input class="search" id="q" type="search" placeholder="⌕ search tasks / metrics / nodes / languages…" aria-label="Search pipelines">
    <button class="chip on" id="viewFlow">Flow</button>
    <button class="chip" id="viewTable">Catalog</button>
    <span class="stats" id="stats"></span>
  </div>
  <nav class="chips" id="chips" aria-label="Task filter">{"".join(chips)}</nav>
  <div class="frame" id="flowFrame">{svg_markup(layout, "light", "app").replace('class="atlas-svg"', 'class="atlas-svg animate"')}</div>
  <div class="frame hidden" id="tblFrame" style="overflow:auto; max-height:70vh"><table class="tbl">
    <thead><tr><th>Task</th><th>Lang</th><th>Metric</th><th>Pipeline ID</th><th>Node chain</th></tr></thead>
    <tbody>{"".join(rows)}</tbody></table></div>
  <footer class="foot">{len(layout["atomic"])} atomic pipelines ({len(layout["threads"])} drawn
  threads; multi-metric pipelines fan out in the scoring lane) ·
  {len(layout["bundles"])} bundle pipelines marked ⛓ in the catalog view ·
  hover to trace a pipeline, click to pin, click a task block or chip to filter</footer>
  <div class="tip" id="tip" role="status"></div>
</div>
<script>const TOTALS = {totals};{APP_JS}</script>
</body>
</html>
"""


def main() -> None:
    layout = build_layout(load_catalog())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        OUT_DIR / "pipeline_atlas.svg": svg_markup(layout, "light", "readme"),
        OUT_DIR / "pipeline_atlas_dark.svg": svg_markup(layout, "dark", "readme"),
        OUT_DIR / "index.html": app_html(layout),
    }
    for path, content in outputs.items():
        path.write_text(content + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(REPO_ROOT)} ({len(content)} bytes)")
    print(f"{len(layout['atomic'])} atomic pipelines, {len(layout['threads'])} threads, "
          f"{len(layout['node_use'])} nodes, {len(layout['bundles'])} bundles")


if __name__ == "__main__":
    main()
