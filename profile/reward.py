#!/usr/bin/env python3
"""
Plot train losses and reward metrics across steps from parsed log data.

Uses parsing functions from parse_perf_logs.py to extract train loss and rollout reward
metrics and creates line plots showing their progression over training steps.
"""

import argparse
import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from parse_perf_logs import parse_train_loss_and_rollout_reward


EVAL_LINE_RE = re.compile(r"eval\s+(\d+):\s*(\{.*\})")


def parse_eval_metrics_from_log(log_path: str):
    """Parse `eval x: {...}` metric dictionaries from a log file.

    Returns:
        Dict mapping metric name -> list of (eval_index, value).
    """
    metrics = {}
    path = Path(log_path)
    if not path.exists():
        return metrics

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = EVAL_LINE_RE.search(line)
            if not match:
                continue
            eval_idx = int(match.group(1))
            dict_str = match.group(2)
            try:
                data = ast.literal_eval(dict_str)
            except Exception:
                # Skip malformed lines
                continue

            for key, value in data.items():
                # Drop duplicate/legacy truncated ratio metric
                if key.endswith("-truncated_ratio"):
                    continue
                # Only plot scalar numeric metrics
                if isinstance(value, (int, float)):
                    metrics.setdefault(key, []).append((eval_idx, float(value)))

    # Ensure each metric series is sorted by eval index
    for key, series in metrics.items():
        series.sort(key=lambda x: x[0])

    return metrics


def plot_train_loss_and_rewards(
    log_path: str,
    output_path: str,
    title: str = "Training Loss and Rewards"
):
    """
    Plot train loss and rollout rewards from a log file.
    
    Args:
        log_path: Path to the log file
        output_path: Path to save the output plot
        title: Title for the plot
    """
    # Parse the log file
    results = parse_train_loss_and_rollout_reward(log_path)
    rollout_rewards = results["rollout_reward"]
    eval_metrics = parse_eval_metrics_from_log(log_path)

    # Pull out main eval score to overlay on rollout plot
    eval_gsm_key = "eval/gsm8k"
    eval_gsm_series = None
    if eval_metrics and eval_gsm_key in eval_metrics:
        eval_gsm_series = eval_metrics[eval_gsm_key]
        # Remove from dict so it doesn't get its own subplot below
        del eval_metrics[eval_gsm_key]
    
    if not rollout_rewards and not eval_metrics and not eval_gsm_series:
        print(f"No rollout reward or eval metric data found in {log_path}")
        return
    
    # Decide how many eval subplots we actually need based on groups
    length_keys = []
    rep_trunc_keys = []
    other_keys = []
    if eval_metrics:
        length_keys = [k for k in eval_metrics.keys() if "response_len" in k]
        rep_trunc_keys = [
            k for k in eval_metrics.keys()
            if k in ("eval/gsm8k/repetition_frac", "eval/gsm8k/truncated_ratio")
        ]
        used_keys = set(length_keys) | set(rep_trunc_keys)
        other_keys = [k for k in eval_metrics.keys() if k not in used_keys]

    # Create a single figure:
    #   subplot 0: rollout raw_reward (+ eval/gsm8k)
    #   remaining subplots: only for non-empty metric groups
    n_subplots = 1
    if length_keys:
        n_subplots += 1
    if rep_trunc_keys:
        n_subplots += 1
    n_subplots += len(other_keys)

    fig, axes = plt.subplots(n_subplots, 1, figsize=(12, 4 * n_subplots), sharex=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    # Plot rollout raw_reward (and eval/gsm8k) in the first subplot
    ax0 = axes[0]
    if rollout_rewards:
        rollout_nums = []
        raw_rewards = []
        
        for rollout_num, reward_data in rollout_rewards:
            rollout_nums.append(rollout_num)
            raw_rewards.append(reward_data.get('raw_reward', None))

        # Plot raw_reward if available
        if any(r is not None for r in raw_rewards):
            ax0.plot(
                rollout_nums,
                raw_rewards,
                'g-',
                linewidth=2,
                marker='s',
                markersize=4,
                label='Raw Reward',
                alpha=0.8,
            )

        # Overlay eval/gsm8k on the same axes if available
        if eval_gsm_series:
            eval_indices, eval_values = zip(*eval_gsm_series)
            ax0.plot(
                eval_indices,
                eval_values,
                'b-o',
                linewidth=2,
                markersize=4,
                label=eval_gsm_key,
                alpha=0.8,
            )

        ax0.set_xlabel('Rollout / Eval Step', fontsize=12)
        ax0.set_ylabel('Reward', fontsize=12)
        ax0.set_title('Rollout Raw Reward and Eval Score Over Steps', fontsize=13, fontweight='bold')
        ax0.grid(True, linestyle='--', alpha=0.7)
        ax0.legend(fontsize=10)

        # Add statistics for raw_reward
        if any(r is not None for r in raw_rewards):
            valid_raw_rewards = [r for r in raw_rewards if r is not None]
            if valid_raw_rewards:
                mean_raw = np.mean(valid_raw_rewards)
                min_raw = np.min(valid_raw_rewards)
                max_raw = np.max(valid_raw_rewards)
                stats_text = f'Raw Reward:\nMin: {min_raw:.4f}\nMax: {max_raw:.4f}\nMean: {mean_raw:.4f}'

                ax0.text(
                    0.02,
                    0.98,
                    stats_text,
                    transform=ax0.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                )
    else:
        ax0.text(
            0.5,
            0.5,
            'No rollout reward data available',
            transform=ax0.transAxes,
            ha='center',
            va='center',
            fontsize=12,
        )
        ax0.set_xlabel('Rollout / Eval Step', fontsize=12)
        ax0.set_ylabel('Reward', fontsize=12)
        ax0.set_title('Rollout Raw Reward and Eval Score Over Steps', fontsize=13, fontweight='bold')

    # Plot eval metrics grouped into subplots (if any)
    if eval_metrics:
        color_cycle = plt.cm.tab10(np.linspace(0, 1, max(1, len(eval_metrics))))
        color_iter = iter(color_cycle)

        ax_idx = 1

        # Length metrics subplot
        if length_keys:
            ax_len = axes[ax_idx]
            for key in length_keys:
                series = eval_metrics[key]
                eval_indices, values = zip(*series)
                color = next(color_iter)
                ax_len.plot(
                    eval_indices,
                    values,
                    '-o',
                    linewidth=2,
                    markersize=4,
                    label=key,
                    color=color,
                    alpha=0.8,
                )

            ax_len.set_xlabel('Eval Index', fontsize=12)
            ax_len.set_ylabel('Response Length', fontsize=12)
            ax_len.set_title('Eval Response Length Metrics', fontsize=13, fontweight='bold')
            ax_len.grid(True, linestyle='--', alpha=0.7)
            ax_len.legend(fontsize=8, loc='best')
            ax_idx += 1

        # repetition_frac and truncated_ratio subplot
        if rep_trunc_keys:
            ax_rep = axes[ax_idx]
            for key in rep_trunc_keys:
                series = eval_metrics[key]
                eval_indices, values = zip(*series)
                color = next(color_iter)
                ax_rep.plot(
                    eval_indices,
                    values,
                    '-o',
                    linewidth=2,
                    markersize=4,
                    label=key,
                    color=color,
                    alpha=0.8,
                )

            ax_rep.set_xlabel('Eval Index', fontsize=12)
            ax_rep.set_ylabel('Fraction', fontsize=12)
            ax_rep.set_title('Eval Repetition and Truncation Fractions', fontsize=13, fontweight='bold')
            ax_rep.grid(True, linestyle='--', alpha=0.7)
            ax_rep.legend(fontsize=8, loc='best')
            ax_idx += 1

        # Any remaining metrics, one subplot each
        for key in other_keys:
            ax_m = axes[ax_idx]
            series = eval_metrics[key]
            eval_indices, values = zip(*series)
            color = next(color_iter)
            ax_m.plot(
                eval_indices,
                values,
                '-o',
                linewidth=2,
                markersize=4,
                label=key,
                color=color,
                alpha=0.8,
            )

            ax_m.set_xlabel('Eval Index', fontsize=12)
            ax_m.set_ylabel(key, fontsize=12)
            ax_m.set_title(f'{key} Over Evaluation Steps', fontsize=13, fontweight='bold')
            ax_m.grid(True, linestyle='--', alpha=0.7)
            ax_m.legend(fontsize=8, loc='best')
            ax_idx += 1

    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_multiple_logs(
    log_files: list,
    output_path: str,
    title: str = "Training Loss and Rewards Comparison"
):
    """
    Plot train loss and rollout rewards from multiple log files for comparison.
    
    Args:
        log_files: List of paths to log files
        output_path: Path to save the output plot
        title: Title for the plot
    """
    all_rollout_rewards = {}
    
    # Parse all log files
    for log_file in log_files:
        log_name = Path(log_file).stem
        results = parse_train_loss_and_rollout_reward(str(log_file))
        all_rollout_rewards[log_name] = results["rollout_reward"]
    
    if not any(all_rollout_rewards.values()):
        print("No rollout reward data found in any log files")
        return
    
    # Create figure with a single subplot for raw_reward comparison
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    # Plot rollout raw_reward from all files
    if any(all_rollout_rewards.values()):
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_rollout_rewards)))
        for (log_name, rollout_rewards), color in zip(all_rollout_rewards.items(), colors):
            if rollout_rewards:
                rollout_nums = []
                raw_rewards = []

                for rollout_num, reward_data in rollout_rewards:
                    rollout_nums.append(rollout_num)
                    raw_rewards.append(reward_data.get('raw_reward', None))

                # Plot raw_reward if available
                if any(r is not None for r in raw_rewards):
                    ax.plot(
                        rollout_nums,
                        raw_rewards,
                        '-',
                        linewidth=2,
                        marker='s',
                        markersize=3,
                        label=f'{log_name} (raw)',
                        color=color,
                        alpha=0.8,
                    )

        ax.set_xlabel('Rollout Number', fontsize=12)
        ax.set_ylabel('Raw Reward', fontsize=12)
        ax.set_title('Rollout Raw Reward Over Steps (Comparison)', fontsize=13, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=9, loc='best')
    else:
        ax.text(
            0.5,
            0.5,
            'No rollout reward data available',
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=12,
        )
        ax.set_xlabel('Rollout Number', fontsize=12)
        ax.set_ylabel('Raw Reward', fontsize=12)
        ax.set_title('Rollout Raw Reward Over Steps (Comparison)', fontsize=13, fontweight='bold')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot train losses and reward metrics from log files.")
    parser.add_argument('--log-file', type=str, default=None,
                        help='Single log file to plot (if not provided, plots all .log files in logs-dir)')
    parser.add_argument('--logs-dir', type=str, default=None,
                        help='Directory containing log files (default: ../logs)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save output plots (default: ./figures)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare multiple log files in a single plot')
    args = parser.parse_args()
    
    # Determine directories
    script_dir = Path(__file__).parent.resolve()
    logs_dir = Path(args.logs_dir) if args.logs_dir else script_dir.parent / 'logs'
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / 'figures'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.log_file:
        # Plot single log file
        log_path = Path(args.log_file)
        if not log_path.exists():
            print(f"Error: Log file not found: {log_path}")
            return
        
        output_path = output_dir / f"{log_path.stem}_rewards.png"
        plot_train_loss_and_rewards(str(log_path), str(output_path))
    else:
        # Find all .log files
        if not logs_dir.exists():
            print(f"Error: Logs directory not found: {logs_dir}")
            return
        
        log_files = sorted(list(logs_dir.glob('*.log')))
        
        if not log_files:
            print(f"No .log files found in {logs_dir}")
            return
        
        if args.compare:
            # Plot all files in comparison mode
            output_path = output_dir / "rewards_comparison.png"
            plot_multiple_logs([str(f) for f in log_files], str(output_path))
        else:
            # Plot each file separately
            for log_file in log_files:
                print(f"\nProcessing: {log_file.name}")
                output_path = output_dir / f"{log_file.stem}_rewards.png"
                plot_train_loss_and_rewards(str(log_file), str(output_path))


if __name__ == "__main__":
    main()
