#!/usr/bin/env python3
"""
Parse performance logs from the logs folder.

Finds lines with "perf N:" pattern containing "MegatronTrainRayActor" or "RolloutManager",
parses the dictionary, averages entries across iterations (excluding first iteration), 
and creates a grouped stacked bar plot.

Each log file becomes a group with 3 stacked bars:
  a) step_time
  b) train_wait_time + train_time
  c) update_weights_time + sleep_time + rollout_time + wake_up_time + log_probs_time + data_preprocess_time + actor_train_time
"""

import re
import ast
import os
import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def parse_perf_lines(log_path: str, skip_first_iter: bool = True) -> dict:
    """
    Parse a log file and extract performance metrics.
    
    Args:
        log_path: Path to the log file
        skip_first_iter: If True, skip the first iteration (iteration 0) from results
    
    Returns a dict of:
        {
            "MegatronTrainRayActor": {key: [values across iterations]},
            "RolloutManager": {key: [values across iterations]}
        }
    """
    # Pattern to match lines with perf N: followed by a dict
    # e.g., "(MegatronTrainRayActor pid=143453) perf 3: {'perf/sleep_time': ...}"
    pattern = r'\((MegatronTrainRayActor|RolloutManager)\s+pid=\d+\).*?perf\s+(\d+):\s*(\{.*\})'
    
    results = {
        "MegatronTrainRayActor": defaultdict(list),
        "RolloutManager": defaultdict(list),
    }
    
    # First pass: find the minimum iteration number
    min_iteration = None
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
            match = re.search(pattern, clean_line)
            if match:
                iteration = int(match.group(2))
                if min_iteration is None or iteration < min_iteration:
                    min_iteration = iteration
    
    # Second pass: collect data, optionally skipping first iteration
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Remove ANSI escape codes
            clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
            
            match = re.search(pattern, clean_line)
            if match:
                actor_type = match.group(1)
                iteration = int(match.group(2))
                dict_str = match.group(3)
                
                # Skip first iteration if requested
                if skip_first_iter and iteration == min_iteration:
                    continue
                
                try:
                    data = ast.literal_eval(dict_str)
                    for key, value in data.items():
                        # Only keep keys ending with "time"
                        if key.endswith('time'):
                            if isinstance(value, (int, float)):
                                results[actor_type][key].append(value)
                except (SyntaxError, ValueError) as e:
                    print(f"Warning: Could not parse dict in line: {line[:100]}... Error: {e}")
                    continue
    
    return results


def compute_averages(results: dict) -> dict:
    """Compute average values for each key."""
    averages = {}
    for actor_type, metrics in results.items():
        averages[actor_type] = {}
        for key, values in metrics.items():
            if values:
                averages[actor_type][key] = np.mean(values)
    return averages


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
    
    # Bar C: update_weights_time + sleep_time + rollout_time + wake_up_time + 
    #        log_probs_time + data_preprocess_time + actor_train_time
    bar_c_components = [
        'update_weights_time', 'sleep_time', 'rollout_time', 
        'wake_up_time', 'log_probs_time', 'data_preprocess_time', 'actor_train_time'
    ]
    bar_c_values = {comp: get_metric_value(averages, comp) for comp in bar_c_components}
    bar_c = sum(bar_c_values.values())
    
    return (bar_a, bar_b, bar_c, bar_b_values, bar_c_values)


def create_grouped_bar_plot(all_data: dict, output_path: str, title: str = "Performance Metrics Comparison"):
    """
    Create a grouped stacked bar plot with individual component fractions labeled.
    
    all_data: dict mapping log_name -> averages dict
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
    
    for log_name in log_names:
        averages = all_data[log_name]
        bar_a, bar_b, bar_c, b_details, c_details = compute_stacked_bar_values(averages)
        bar_a_values.append(bar_a)
        bar_b_details.append(b_details)
        bar_c_details.append(c_details)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(max(12, n_groups * 4), 10))
    
    x = np.arange(n_groups)
    width = 0.25
    
    # Color palettes for each bar type
    colors_a = ['#e74c3c']  # Red for step_time
    colors_b = ['#3498db', '#2980b9']  # Blues for bar B components
    colors_c = ['#2ecc71', '#27ae60', '#1abc9c', '#16a085', '#f39c12', '#e67e22', '#d35400']  # Greens/oranges for bar C
    
    # Bar A: step_time (single component)
    bars_a = ax.bar(x - width, bar_a_values, width, color=colors_a[0], alpha=0.85)
    
    # Add labels for bar A (100% since it's single component)
    for i, (bar, val) in enumerate(zip(bars_a, bar_a_values)):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                    f'step\n{val:.2f}s\n(100%)', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Bar B: stacked components (train_wait_time + train_time)
    bar_b_components = ['train_wait_time', 'train_time']
    bar_b_short_names = ['train_wait', 'train']
    
    # Calculate totals for bar B for percentage
    bar_b_totals = [sum(bar_b_details[i].values()) for i in range(n_groups)]
    
    bottom_b = np.zeros(n_groups)
    for j, comp in enumerate(bar_b_components):
        values = [bar_b_details[i].get(comp, 0) for i in range(n_groups)]
        bars = ax.bar(x, values, width, bottom=bottom_b, color=colors_b[j % len(colors_b)], alpha=0.85)
        
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
        'update_weights_time', 'sleep_time', 'rollout_time', 
        'wake_up_time', 'log_probs_time', 'data_preprocess_time', 'actor_train_time'
    ]
    bar_c_short_names = ['upd_wts', 'sleep', 'rollout', 'wake_up', 'log_probs', 'data_prep', 'actor_train']
    
    # Calculate totals for bar C for percentage
    bar_c_totals = [sum(bar_c_details[i].values()) for i in range(n_groups)]
    
    bottom_c = np.zeros(n_groups)
    for j, comp in enumerate(bar_c_components):
        values = [bar_c_details[i].get(comp, 0) for i in range(n_groups)]
        bars = ax.bar(x + width, values, width, bottom=bottom_c, color=colors_c[j % len(colors_c)], alpha=0.85)
        
        # Add labels inside each stack segment with percentage
        for i, (bar, val) in enumerate(zip(bars, values)):
            if val > 0.3:  # Only label if value is significant enough to fit text
                pct = (val / bar_c_totals[i] * 100) if bar_c_totals[i] > 0 else 0
                ax.text(bar.get_x() + bar.get_width() / 2, bottom_c[i] + val / 2,
                        f'{bar_c_short_names[j]}\n{val:.2f}s\n({pct:.0f}%)', ha='center', va='center', 
                        fontsize=6, fontweight='bold', color='white')
        bottom_c += values
    
    # Add total labels on top of each bar
    for i in range(n_groups):
        # Total for bar A
        ax.text(x[i] - width, bar_a_values[i] + 0.3, f'Σ{bar_a_values[i]:.2f}s', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
        # Total for bar B
        total_b = sum(bar_b_details[i].values())
        ax.text(x[i], total_b + 0.3, f'Σ{total_b:.2f}s', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
        # Total for bar C
        total_c = sum(bar_c_details[i].values())
        ax.text(x[i] + width, total_c + 0.3, f'Σ{total_c:.2f}s', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
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
        print(f"{log_name}:\n  Bar B: {b_parts}\n  Bar C: {c_parts}")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Parse performance logs and create bar plots.")
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
    
    # Collect data from all log files
    all_data = {}
    
    for log_file in log_files:
        print(f"\nProcessing: {log_file.name}")
        
        # Parse the log file
        results = parse_perf_lines(str(log_file), skip_first_iter=skip_first)
        
        # Print summary of found metrics
        for actor_type, metrics in results.items():
            if metrics:
                print(f"  {actor_type}:")
                for key, values in metrics.items():
                    print(f"    {key}: {len(values)} entries, avg={np.mean(values):.4f}")
        
        # Compute averages
        averages = compute_averages(results)
        all_data[log_file.stem] = averages
    
    # Create grouped bar plot
    output_path = output_dir / "perf_comparison.png"
    title = "Performance Metrics Comparison (Excluding First Iteration)" if skip_first else "Performance Metrics Comparison"
    create_grouped_bar_plot(all_data, str(output_path), title)


if __name__ == "__main__":
    main()
