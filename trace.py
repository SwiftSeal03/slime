#!/usr/bin/env python3
"""
Visualize timestamp trace from a directory (e.g. from --timestamp-path) with 5 tracks:
  iteration, rollout, training, backup, weight update.
Each begin/end pair is drawn as a horizontal bar with duration shown.
Reads <timestamp_path>/main.txt, <timestamp_path>/rollout.txt, <timestamp_path>/actor-*.txt and merges by time.
Weight-update events are emitted from the update_weight modules (actor-*.txt); for multiple ranks
the first-seen begin and last-seen end per rollout_id are used.
Optional weight_updates_gather_end and weight_updates_offload_end (when present) subdivide the
weight-update bar into 2–3 segments; if absent, a single begin→end bar is drawn.
Usage: python trace.py <timestamp_path> [--output timestamp/<same_name>.png]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Fixed process files and glob for actor ranks (must match slime.utils.timestamp)
TRACE_PROCESS_FILES = ("main.txt", "rollout.txt")
TRACE_ACTOR_GLOB = "actor-*.txt"


def parse_timestamp_file(path: Path) -> list[tuple[float, str]]:
    """Parse lines 'timestamp\\tmessage' into (ts, msg) pairs."""
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            try:
                ts = float(parts[0])
            except ValueError:
                continue
            events.append((ts, parts[1].strip()))
    return events


def load_events_from_timestamp_dir(path: Path) -> list[tuple[float, str]]:
    """Load and merge events from <path>/main.txt, <path>/rollout.txt, <path>/actor-*.txt."""
    events: list[tuple[float, str]] = []
    for name in TRACE_PROCESS_FILES:
        fpath = path / name
        if fpath.exists():
            events.extend(parse_timestamp_file(fpath))
    for fpath in sorted(path.glob(TRACE_ACTOR_GLOB)):
        events.extend(parse_timestamp_file(fpath))
    events.sort(key=lambda x: x[0])
    return events


def extract_key(msg: str, prefix: str) -> str | None:
    """Extract key from message for matching begin/end. E.g. 'iteration_begin 0' -> '0'."""
    if not msg.startswith(prefix):
        return None
    rest = msg[len(prefix) :].strip()
    return rest if rest else "_"  # no suffix -> "_"


def pair_intervals(
    events: list[tuple[float, str]],
    begin_prefix: str,
    end_prefix: str,
) -> list[tuple[float, float, str]]:
    """Match begin/end by key. For duplicate keys (e.g. multiple ranks): use first-seen start, last-seen end."""
    begins: dict[str, float] = {}  # key -> min(ts) over all begins
    ends: dict[str, float] = {}     # key -> max(ts) over all ends
    for ts, msg in events:
        begin_key = extract_key(msg, begin_prefix)
        end_key = extract_key(msg, end_prefix)
        if begin_key is not None:
            begins[begin_key] = min(begins[begin_key], ts) if begin_key in begins else ts
        elif end_key is not None:
            ends[end_key] = max(ends[end_key], ts) if end_key in ends else ts
    # One interval per key that has both begin and end
    keys_with_both = sorted(begins.keys() & ends.keys())
    return [(begins[k], ends[k], k) for k in keys_with_both]


def get_weight_update_mid_events(
    events: list[tuple[float, str]],
) -> dict[str, tuple[float | None, float | None]]:
    """Per-key (gather_end_ts, offload_end_ts) for weight update; None if event missing. Uses min/max over ranks."""
    gather: dict[str, float] = {}  # key -> max(ts)
    offload: dict[str, float] = {}  # key -> max(ts)
    for ts, msg in events:
        k = extract_key(msg, "weight_updates_gather_end ")
        if k is not None:
            gather[k] = max(gather[k], ts) if k in gather else ts
            continue
        k = extract_key(msg, "weight_updates_offload_end ")
        if k is not None:
            offload[k] = max(offload[k], ts) if k in offload else ts
    keys = sorted(set(gather.keys()) | set(offload.keys()))
    return {k: (gather.get(k), offload.get(k)) for k in keys}


def weight_update_segments(
    start: float,
    end: float,
    gather_end: float | None,
    offload_end: float | None,
) -> list[tuple[float, float]]:
    """Split [start, end] into 1–3 segments when gather_end and/or offload_end are present (and within range)."""
    if gather_end is None and offload_end is None:
        return [(start, end)]
    # Clamp mid points to [start, end] and ensure order
    g = gather_end if gather_end is not None and start <= gather_end <= end else None
    o = offload_end if offload_end is not None and start <= offload_end <= end else None
    if g is not None and o is not None and g > o:
        g, o = o, g  # ensure gather <= offload
    if g is None and o is None:
        return [(start, end)]
    if g is not None and o is None:
        return [(start, g), (g, end)]
    if g is None and o is not None:
        return [(start, o), (o, end)]
    return [(start, g), (g, o), (o, end)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize timestamp trace directory with 5 tracks.")
    parser.add_argument(
        "timestamp_path",
        type=Path,
        help="Directory containing main.txt, rollout.txt, actor-*.txt (from --timestamp-path).",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output image path (default: timestamp/<timestamp_path_name>.png)",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    args = parser.parse_args()

    trace_dir = args.timestamp_path.resolve()
    if not trace_dir.is_dir():
        raise SystemExit(f"Not a directory: {trace_dir}")

    if args.output is None:
        default_dir = Path("timestamp")
        default_dir.mkdir(parents=True, exist_ok=True)
        args.output = default_dir / f"{trace_dir.name}.png"

    events = load_events_from_timestamp_dir(trace_dir)
    if not events:
        raise SystemExit("No events parsed from timestamp directory.")

    t0 = min(ts for ts, _ in events)
    # Normalize to seconds from first event
    def norm(ts: float) -> float:
        return ts - t0

    tracks = [
        ("iteration", "iteration_begin", "iteration_end"),
        ("rollout", "rollout_begin", "rollout_end"),
        ("training", "training_begin", "training_end"),
        # ("backup", "backup_begin", "backup_end"),
        ("weight update", "weight_updates_begin", "weight_updates_end"),
    ]

    track_intervals: list[list[tuple[float, float, str]]] = []
    for _, begin_prefix, end_prefix in tracks:
        track_intervals.append(pair_intervals(events, begin_prefix, end_prefix))

    weight_update_mid = get_weight_update_mid_events(events)

    n_tracks = len(tracks)
    fig, axes = plt.subplots(n_tracks, 1, figsize=(14, 5), sharex=True, gridspec_kw={"height_ratios": [1] * n_tracks})
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]  # backup color "#f39c12" omitted

    x_max = norm(events[-1][0]) if events else 1

    for ax, track_name, intervals, color in zip(axes, [t[0] for t in tracks], track_intervals, colors):
        # Average bar length excluding first and last bar
        avg_len = None
        if len(intervals) >= 3:
            lengths = [end - start for start, end, _ in intervals[1:-1]]
            avg_len = sum(lengths) / len(lengths)
        ax.set_ylabel(track_name, fontsize=10)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([0])
        ax.set_yticklabels([])
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for start, end, key in intervals:
            if track_name == "weight update":
                gather_end, offload_end = weight_update_mid.get(key, (None, None))
                segs = weight_update_segments(start, end, gather_end, offload_end)
                segment_alphas = [0.95, 0.8, 0.65][: len(segs)]  # darker = earlier phase
                for seg_idx, (s_start, s_end) in enumerate(segs):
                    width = max(s_end - s_start, 1e-6)
                    bar = mpatches.FancyBboxPatch(
                        (norm(s_start), -0.4),
                        width,
                        0.8,
                        boxstyle="round,pad=0.01,rounding_size=0.02",
                        facecolor=color,
                        edgecolor="none",
                        alpha=segment_alphas[seg_idx],
                    )
                    ax.add_patch(bar)
                    duration = s_end - s_start
                    label = f"{duration:.1f}s"
                    ax.text(
                        norm(s_start) + width / 2,
                        0,
                        label,
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white",
                        weight="bold",
                    )
            else:
                width = max(end - start, 1e-6)  # avoid zero-width
                bar = mpatches.FancyBboxPatch(
                    (norm(start), -0.4),
                    width,
                    0.8,
                    boxstyle="round,pad=0.01,rounding_size=0.02",
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.85,
                )
                ax.add_patch(bar)
                duration = end - start
                label = f"{duration:.1f}s"
                ax.text(
                    norm(start) + width / 2,
                    0,
                    label,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                    weight="bold",
                )
        # Show average bar length at the end of the track (horizontal)
        if avg_len is not None:
            ax.text(
                x_max,
                0,
                f"avg: {avg_len:.1f}s",
                ha="right",
                va="center",
                fontsize=11,
                color="black",
            )
        ax.set_xlim(0, x_max)
        ax.grid(axis="x", alpha=0.3)

    axes[-1].set_xlabel("Time (s from first event)")
    plt.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
