#!/usr/bin/env python3
"""
Create performance breakdown plots from parsed log data.

Uses parsing functions from parse_perf_logs.py to extract performance metrics
and creates grouped stacked bar plots showing time breakdowns.
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from parse_perf_logs import parse_perf_lines, parse_real_perf_lines, compute_averages, find_eval_iterations


def get_metric_value(averages: dict, metric_name: str) -> float:
    """Get the average value of a metric from either actor type."""
    # Remove 'perf/' prefix if present for comparison
    search_keys = [metric_name, f'perf/{metric_name}']
    
    for actor_type in averages:
        for key in search_keys:
            if key in averages[actor_type]:
                return averages[actor_type][key]
    return 0.0


def compute_stacked_bar_values(averages: dict) -> tuple:
    """
    Compute the values for the 3 stacked bars.
    
    Returns (bar_a, bar_b, bar_c) where:
        bar_a: step_time
        bar_b: train_wait_time + train_time
        bar_c: update_weights_time + sleep_time + rollout_time + wake_up_time + 
               log_probs_time + data_preprocess_time + actor_train_time
    """
    # Bar A: step_time
    bar_a = get_metric_value(averages, 'step_time')
    
    # Bar B: train_wait_time + train_time
    bar_b_components = ['train_wait_time', 'train_time']
    bar_b_values = {comp: get_metric_value(averages, comp) for comp in bar_b_components}
    bar_b = sum(bar_b_values.values())
    
    # Bar C: actor_train_time + log_probs_time + update_weights_time + sleep_time + rollout_time + 
    #        wake_up_time + data_preprocess_time
    bar_c_components = [
        'actor_train_time', 'log_probs_time', 'update_weights_time', 'sleep_time', 'rollout_time', 
        'wake_up_time', 'data_preprocess_time'
    ]
    bar_c_values = {comp: get_metric_value(averages, comp) for comp in bar_c_components}
    bar_c = sum(bar_c_values.values())
    
    return (bar_a, bar_b, bar_c, bar_b_values, bar_c_values)


def compute_real_perf_bar_values(real_perf_averages: dict) -> tuple:
    """
    Compute the values for bar D from real-perf metrics.
    
    Returns (bar_d, bar_d_values) where:
        bar_d: all real-perf timing components stacked
    """
    # Bar D: all real-perf metrics from train.py
    bar_d_components = [
        'actor_train_time', 'generate_time', 'offload_rollout_time',
        'critic_train_launch_time', 'critic_train_wait_time', 'save_time',
        'offload_train_time', 'onload_rollout_time', 'update_weights_time',
        'onload_rollout_additional_time'
    ]
    bar_d_values = {comp: real_perf_averages.get(comp, 0.0) for comp in bar_d_components}
    bar_d = sum(bar_d_values.values())
    
    return (bar_d, bar_d_values)


def create_grouped_bar_plot(all_data: dict, all_real_perf_data: dict, output_path: str, title: str = "Performance Metrics Comparison"):
    """
    Create a grouped stacked bar plot with individual component fractions labeled.
    
    all_data: dict mapping log_name -> averages dict
    all_real_perf_data: dict mapping log_name -> real_perf averages dict
    """
    log_names = list(all_data.keys())
    n_groups = len(log_names)
    
    if n_groups == 0:
        print("No data to plot.")
        return
    
    # Compute stacked bar values for each log file
    bar_a_values = []
    bar_b_details = []
    bar_c_details = []
    bar_d_details = []
    
    for log_name in log_names:
        averages = all_data[log_name]
        bar_a, bar_b, bar_c, b_details, c_details = compute_stacked_bar_values(averages)
        bar_a_values.append(bar_a)
        bar_b_details.append(b_details)
        bar_c_details.append(c_details)
        
        # Compute bar D from real-perf data if available
        if log_name in all_real_perf_data:
            bar_d, d_details = compute_real_perf_bar_values(all_real_perf_data[log_name])
            bar_d_details.append(d_details)
        else:
            bar_d_details.append({})
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(max(12, n_groups * 4), 10))
    
    x = np.arange(n_groups)
    width = 0.2
    
    # Color palettes for each bar type
    colors_a = ['#e74c3c']  # Red for step_time
    colors_b = ['#3498db', '#2980b9']  # Blues for bar B components
    colors_c = ['#2ecc71', '#27ae60', '#1abc9c', '#16a085', '#f39c12', '#e67e22', '#d35400']  # Greens/oranges for bar C
    colors_d = ['#6a1b9a', '#7b1fa2', '#8e24aa', '#9c27b0', '#ab47bc', '#ba68c8', '#ce93d8', '#e1bee7', '#f3e5f5', '#ede7f6']  # Darker purples for bar D with higher contrast
    
    # Bar A: step_time (single component)
    bars_a = ax.bar(x - width * 1.5, bar_a_values, width, color=colors_a[0], alpha=0.85)
    
    # Add labels for bar A (100% since it's single component)
    for i, (bar, val) in enumerate(zip(bars_a, bar_a_values)):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                    f'step\n{val:.2f}s\n(100%)', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Bar B: stacked components (train_time + train_wait_time)
    bar_b_components = ['train_time', 'train_wait_time']
    bar_b_short_names = ['train', 'train_wait']
    
    # Calculate totals for bar B for percentage
    bar_b_totals = [sum(bar_b_details[i].values()) for i in range(n_groups)]
    
    bottom_b = np.zeros(n_groups)
    for j, comp in enumerate(bar_b_components):
        values = [bar_b_details[i].get(comp, 0) for i in range(n_groups)]
        bars = ax.bar(x - width * 0.5, values, width, bottom=bottom_b, color=colors_b[j % len(colors_b)], alpha=0.85)
        
        # Add labels inside each stack segment with percentage
        for i, (bar, val) in enumerate(zip(bars, values)):
            if val > 0.1:  # Only label if value is significant
                pct = (val / bar_b_totals[i] * 100) if bar_b_totals[i] > 0 else 0
                ax.text(bar.get_x() + bar.get_width() / 2, bottom_b[i] + val / 2,
                        f'{bar_b_short_names[j]}\n{val:.2f}s\n({pct:.0f}%)', ha='center', va='center', 
                        fontsize=7, fontweight='bold', color='white')
        bottom_b += values
    
    # Bar C: stacked components
    bar_c_components = [
        'actor_train_time', 'log_probs_time', 'update_weights_time', 'sleep_time', 'rollout_time', 
        'wake_up_time', 'data_preprocess_time'
    ]
    bar_c_short_names = ['actor_train', 'log_probs', 'upd_wts', 'sleep', 'rollout', 'wake_up', 'data_prep']
    
    # Calculate totals for bar C for percentage
    bar_c_totals = [sum(bar_c_details[i].values()) for i in range(n_groups)]
    
    bottom_c = np.zeros(n_groups)
    for j, comp in enumerate(bar_c_components):
        values = [bar_c_details[i].get(comp, 0) for i in range(n_groups)]
        bars = ax.bar(x + width * 0.5, values, width, bottom=bottom_c, color=colors_c[j % len(colors_c)], alpha=0.85)
        
        # Add labels inside each stack segment with percentage
        for i, (bar, val) in enumerate(zip(bars, values)):
            if val > 0.3:  # Only label if value is significant enough to fit text
                pct = (val / bar_c_totals[i] * 100) if bar_c_totals[i] > 0 else 0
                ax.text(bar.get_x() + bar.get_width() / 2, bottom_c[i] + val / 2,
                        f'{bar_c_short_names[j]}\n{val:.2f}s\n({pct:.0f}%)', ha='center', va='center', 
                        fontsize=6, fontweight='bold', color='white')
        bottom_c += values
    
    # Bar D: real-perf stacked components
    bar_d_components = [
        'actor_train_time', 'generate_time', 'offload_rollout_time',
        'critic_train_launch_time', 'critic_train_wait_time', 'save_time',
        'offload_train_time', 'onload_rollout_time', 'update_weights_time',
        'onload_rollout_additional_time'
    ]
    bar_d_short_names = ['actor', 'gen', 'off_roll', 'critic_launch', 'critic_wait', 'save',
                         'off_train', 'on_roll', 'upd_wts', 'on_roll_add']
    
    # Calculate totals for bar D for percentage
    bar_d_totals = [sum(bar_d_details[i].values()) for i in range(n_groups)]
    
    bottom_d = np.zeros(n_groups)
    for j, comp in enumerate(bar_d_components):
        values = [bar_d_details[i].get(comp, 0) for i in range(n_groups)]
        bars = ax.bar(x + width * 1.5, values, width, bottom=bottom_d, color=colors_d[j % len(colors_d)], alpha=0.85)
        
        # Add labels inside each stack segment with percentage
        for i, (bar, val) in enumerate(zip(bars, values)):
            if val > 0.3:  # Only label if value is significant enough to fit text
                pct = (val / bar_d_totals[i] * 100) if bar_d_totals[i] > 0 else 0
                ax.text(bar.get_x() + bar.get_width() / 2, bottom_d[i] + val / 2,
                        f'{bar_d_short_names[j]}\n{val:.2f}s\n({pct:.0f}%)', ha='center', va='center', 
                        fontsize=6, fontweight='bold', color='white')
        bottom_d += values
    
    # Add total labels on top of each bar
    for i in range(n_groups):
        # Total for bar A
        ax.text(x[i] - width * 1.5, bar_a_values[i] + 0.3, f'Σ{bar_a_values[i]:.2f}s', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
        # Total for bar B
        total_b = sum(bar_b_details[i].values())
        ax.text(x[i] - width * 0.5, total_b + 0.3, f'Σ{total_b:.2f}s', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
        # Total for bar C
        total_c = sum(bar_c_details[i].values())
        ax.text(x[i] + width * 0.5, total_c + 0.3, f'Σ{total_c:.2f}s', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
        # Total for bar D
        total_d = sum(bar_d_details[i].values())
        if total_d > 0:
            ax.text(x[i] + width * 1.5, total_d + 0.3, f'Σ{total_d:.2f}s', 
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Calculate maximum bar height including top labels
    max_bar_height = 0
    for i in range(n_groups):
        max_bar_height = max(max_bar_height, bar_a_values[i])
        max_bar_height = max(max_bar_height, sum(bar_b_details[i].values()))
        max_bar_height = max(max_bar_height, sum(bar_c_details[i].values()))
        total_d = sum(bar_d_details[i].values())
        if total_d > 0:
            max_bar_height = max(max_bar_height, total_d)
    
    # Set y-axis limit with padding for top labels (0.3 offset + some extra space)
    ax.set_ylim(0, max_bar_height * 1.15 + 1.0)
    
    # Customize the plot
    ax.set_xlabel('Log File', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(log_names, rotation=0, ha='center', fontsize=11)
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Print breakdown to console
    print("\nBreakdown details:")
    for i, log_name in enumerate(log_names):
        b_parts = ', '.join([f"{k.replace('_time', '')}={v:.2f}" for k, v in bar_b_details[i].items() if v > 0])
        c_parts = ', '.join([f"{k.replace('_time', '')}={v:.2f}" for k, v in bar_c_details[i].items() if v > 0])
        d_parts = ', '.join([f"{k.replace('_time', '')}={v:.2f}" for k, v in bar_d_details[i].items() if v > 0])
        print(f"{log_name}:\n  Bar B: {b_parts}\n  Bar C: {c_parts}\n  Bar D: {d_parts}")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Parse performance logs and create breakdown plots.")
    parser.add_argument('--logs-dir', type=str, default=None,
                        help='Directory containing log files (default: ../logs)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save output plots (default: ./figures)')
    parser.add_argument('--include-first-iter', action='store_true',
                        help='Include the first iteration in average calculation (excluded by default)')
    args = parser.parse_args()
    
    # Determine directories
    script_dir = Path(__file__).parent.resolve()
    logs_dir = Path(args.logs_dir) if args.logs_dir else script_dir.parent / 'logs'
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / 'figures'
    
    skip_first = not args.include_first_iter
    
    if not logs_dir.exists():
        print(f"Error: Logs directory not found: {logs_dir}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .log files
    log_files = sorted(list(logs_dir.glob('*.log')))
    
    if not log_files:
        print(f"No .log files found in {logs_dir}")
        return
    
    print(f"Found {len(log_files)} log file(s) in {logs_dir}")
    if skip_first:
        print("Note: First iteration is excluded from average calculation")
    print("Note: Iterations with evaluations and iterations immediately after evaluations are excluded from average calculation")
    
    # Collect data from all log files
    all_data = {}
    all_real_perf_data = {}
    
    for log_file in log_files:
        print(f"\nProcessing: {log_file.name}")
        
        # Find eval iterations to skip
        eval_iters = find_eval_iterations(str(log_file))
        if eval_iters:
            print(f"  Found {len(eval_iters)} eval iteration(s): {sorted(eval_iters)}")
        
        # Also skip iterations immediately after evaluations (since train_wait_time depends on previous iteration)
        skip_iters = set()
        if eval_iters:
            skip_iters.update(eval_iters)
            # Add eval_iter + 1 for each eval iteration
            for eval_iter in eval_iters:
                skip_iters.add(eval_iter + 1)
            if skip_iters:
                print(f"  Skipping eval iterations and next iterations: {sorted(skip_iters)}")
        
        # Determine which iterations to skip
        # If skip_first is True, also skip the first iteration
        if skip_first:
            # Find first perf iteration
            pattern = r'perf\s+(\d+):'
            min_iter = None
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
                    match = re.search(pattern, clean_line)
                    if match:
                        iter_num = int(match.group(1))
                        if min_iter is None or iter_num < min_iter:
                            min_iter = iter_num
                        if min_iter is not None:
                            break  # Found first, can stop
            
            if min_iter is not None:
                skip_iters.add(min_iter)
        
        if not skip_iters:
            skip_iters = False
        
        if skip_iters and isinstance(skip_iters, set) and skip_iters:
            print(f"  Total iterations to skip: {sorted(skip_iters)}")
        
        # Parse the log file for perf lines
        results = parse_perf_lines(str(log_file), skip_iterations=skip_iters)
        
        # Print summary of found metrics
        for actor_type, metrics in results.items():
            if metrics:
                print(f"  {actor_type}:")
                for key, values in metrics.items():
                    print(f"    {key}: {len(values)} entries, avg={np.mean(values):.4f}")
        
        # Compute averages
        averages = compute_averages(results)
        all_data[log_file.stem] = averages
        
        # Parse real-perf lines (use same skip logic)
        real_perf_results = parse_real_perf_lines(str(log_file), skip_iterations=skip_iters)
        if real_perf_results.get("real_perf"):
            real_perf_averages = {}
            for key, values in real_perf_results["real_perf"].items():
                if values:
                    real_perf_averages[key] = np.mean(values)
            all_real_perf_data[log_file.stem] = real_perf_averages
            print(f"  Real-perf: {len(real_perf_averages)} metrics found")
    
    # Create grouped bar plot
    output_path = output_dir / "perf_comparison.png"
    title_parts = []
    if skip_first:
        title_parts.append("Excluding First Iteration")
    title_parts.append("Excluding Eval Iterations and Next Iterations")
    title = "Performance Metrics Comparison"
    if title_parts:
        title += f" ({', '.join(title_parts)})"
    create_grouped_bar_plot(all_data, all_real_perf_data, str(output_path), title)


if __name__ == "__main__":
    main()
