#!/usr/bin/env python3
"""
Parse performance logs from log files.

Provides functions to parse different types of log entries:
- Performance metrics (perf N:)
- Real performance metrics (real-perf N:)
- Training loss (step N:)
- Rollout rewards (rollout N:)
"""

import re
import ast
from collections import defaultdict
from typing import List, Tuple, Optional, Callable, Union, Set

import numpy as np


def parse_log_with_prefix(
    log_path: str,
    prefix_name: str,
    pattern_modifier: Optional[str] = None,
    extract_iteration: Optional[Callable] = None,
    filter_dict: Optional[Callable] = None,
    skip_iterations: Union[bool, Set[int], List[int]] = False
) -> List[Tuple[int, dict]]:
    """
    Generic function to parse log lines with pattern "<prefix_name>: {...}" or similar.
    
    Args:
        log_path: Path to the log file
        prefix_name: The prefix to match (e.g., "real-perf", "step", "rollout")
        pattern_modifier: Optional regex pattern to add before the prefix (e.g., for actor types)
        extract_iteration: Optional function to extract iteration number from match groups.
                          If None, assumes first group is iteration number.
        filter_dict: Optional function to filter/transform the parsed dictionary.
                     Signature: filter_dict(data: dict, match: re.Match) -> dict or None
        skip_iterations: If True, skip the first iteration (iteration 0) from results.
                        If a set/list of ints, skip those specific iteration numbers.
                        For backward compatibility, True is treated as skip_first_iter=True.
    
    Returns:
        List of tuples: [(iteration_num, dict_data), ...]
    """
    # Build pattern: if pattern_modifier is provided, include it before prefix
    if pattern_modifier:
        # For patterns like "(MegatronTrainRayActor pid=123) perf N: {...}"
        pattern = rf'{pattern_modifier}.*?{prefix_name}\s+(\d+):\s*(\{{.*?\}})'
    else:
        # For patterns like "real-perf N: {...}" or "step N: {...}"
        pattern = rf'{prefix_name}\s+(\d+):\s*(\{{.*?\}})'
    
    results = []
    
    # Normalize skip_iterations: convert bool/True to set of first iteration, or ensure it's a set
    skip_set: Set[int] = set()
    if skip_iterations is True:
        # Backward compatibility: find first iteration and skip it
        min_iteration = None
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
                match = re.search(pattern, clean_line)
                if match:
                    if extract_iteration:
                        iteration = extract_iteration(match)
                    else:
                        iteration = int(match.group(1))
                    if min_iteration is None or iteration < min_iteration:
                        min_iteration = iteration
        if min_iteration is not None:
            skip_set.add(min_iteration)
    elif skip_iterations:
        # Convert list/set to set
        skip_set = set(skip_iterations)
    
    # Second pass: collect data
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Remove ANSI escape codes
            clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
            
            match = re.search(pattern, clean_line)
            if match:
                if extract_iteration:
                    iteration = extract_iteration(match)
                else:
                    iteration = int(match.group(1))
                # Get the last group which always contains the dictionary
                dict_str = match.groups()[-1]
                
                # Skip iteration if in skip_set
                if iteration in skip_set:
                    continue
                
                try:
                    data = ast.literal_eval(dict_str)
                    
                    # Apply filter if provided
                    if filter_dict:
                        data = filter_dict(data, match)
                        if data is None:
                            continue
                    
                    results.append((iteration, data))
                except (SyntaxError, ValueError) as e:
                    print(f"Warning: Could not parse dict in line: {line[:100]}... Error: {e}")
                    continue
    
    return results


def parse_real_perf_lines(log_path: str, skip_iterations: Union[bool, Set[int], List[int]] = True) -> dict:
    """
    Parse a log file and extract real-perf performance metrics from train.py.
    
    Args:
        log_path: Path to the log file
        skip_iterations: If True, skip the first iteration (iteration 0) from results.
                        If a set/list of ints, skip those specific iteration numbers.
    
    Returns a dict of:
        {"real_perf": {key: [values across iterations]}}
    """
    def filter_time_keys(data: dict, match) -> dict:
        """Filter to only keep keys ending with 'time' and numeric values."""
        filtered = {}
        for key, value in data.items():
            if key.endswith('time') and isinstance(value, (int, float)):
                filtered[key] = value
        return filtered if filtered else None
    
    parsed_data = parse_log_with_prefix(
        log_path=log_path,
        prefix_name="real-perf",
        filter_dict=filter_time_keys,
        skip_iterations=skip_iterations
    )
    
    results = defaultdict(list)
    for iteration, data in parsed_data:
        for key, value in data.items():
            results[key].append(value)
    
    return {"real_perf": dict(results)}


def parse_perf_lines(log_path: str, skip_iterations: Union[bool, Set[int], List[int]] = True) -> dict:
    """
    Parse a log file and extract performance metrics.
    
    Args:
        log_path: Path to the log file
        skip_iterations: If True, skip the first iteration (iteration 0) from results.
                        If a set/list of ints, skip those specific iteration numbers.
    
    Returns a dict of:
        {
            "MegatronTrainRayActor": {key: [values across iterations]},
            "RolloutManager": {key: [values across iterations]}
        }
    """
    def extract_iteration_from_match(match) -> int:
        """Extract iteration number from match (group 2 when pattern_modifier is used)."""
        return int(match.group(2))
    
    def filter_time_keys_with_actor(data: dict, match) -> dict:
        """Filter to only keep keys ending with 'time' and add actor_type to the dict."""
        actor_type = match.group(1)
        filtered = {}
        for key, value in data.items():
            if key.endswith('time') and isinstance(value, (int, float)):
                filtered[key] = value
        # Store actor_type in the dict for later separation
        if filtered:
            filtered['_actor_type'] = actor_type
            return filtered
        return None
    
    # Use pattern_modifier to include actor type before "perf"
    pattern_modifier = r'\((MegatronTrainRayActor|RolloutManager)\s+pid=\d+\)'
    
    parsed_data = parse_log_with_prefix(
        log_path=log_path,
        prefix_name="perf",
        pattern_modifier=pattern_modifier,
        extract_iteration=extract_iteration_from_match,
        filter_dict=filter_time_keys_with_actor,
        skip_iterations=skip_iterations
    )
    
    results = {
        "MegatronTrainRayActor": defaultdict(list),
        "RolloutManager": defaultdict(list),
    }
    
    # Separate data by actor type
    for iteration, data in parsed_data:
        actor_type = data.pop('_actor_type', None)
        if actor_type and actor_type in results:
            for key, value in data.items():
                results[actor_type][key].append(value)
    
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


def parse_train_loss_and_rollout_reward(log_path: str) -> dict:
    """
    Parse train loss and rollout reward from log file.
    
    Args:
        log_path: Path to the log file
    
    Returns a dict of:
        {
            "train_loss": [(step, loss_value), ...],
            "rollout_reward": [(rollout_num, {"raw_reward": ..., "rewards": ...}), ...]
        }
    """
    def filter_train_loss(data: dict, match) -> dict:
        """Filter to only keep train/loss."""
        if 'train/loss' in data:
            return {'train/loss': data['train/loss']}
        return None
    
    def filter_rollout_reward(data: dict, match) -> dict:
        """Filter to only keep rollout reward fields."""
        reward_data = {}
        if 'rollout/raw_reward' in data:
            reward_data['raw_reward'] = data['rollout/raw_reward']
        if 'rollout/rewards' in data:
            reward_data['rewards'] = data['rollout/rewards']
        return reward_data if reward_data else None
    
    # Parse step lines for train loss
    step_data = parse_log_with_prefix(
        log_path=log_path,
        prefix_name="step",
        filter_dict=filter_train_loss,
        skip_first_iter=False
    )
    
    # Parse rollout lines for rewards
    rollout_data = parse_log_with_prefix(
        log_path=log_path,
        prefix_name="rollout",
        filter_dict=filter_rollout_reward,
        skip_first_iter=False
    )
    
    # Format results
    train_losses = [(step_num, data['train/loss']) for step_num, data in step_data]
    rollout_rewards = [(rollout_num, data) for rollout_num, data in rollout_data]
    
    return {
        "train_loss": train_losses,
        "rollout_reward": rollout_rewards
    }


def find_eval_iterations(log_path: str) -> Set[int]:
    """
    Find all iteration numbers that have evaluations.
    
    Evaluations are logged as "eval N: {...}" where N is the rollout_id/iteration number.
    
    Args:
        log_path: Path to the log file
    
    Returns:
        Set of iteration numbers that have evaluations
    """
    eval_iterations = set()
    pattern = r'eval\s+(\d+):'
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Remove ANSI escape codes
            clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
            match = re.search(pattern, clean_line)
            if match:
                eval_iter = int(match.group(1))
                eval_iterations.add(eval_iter)
    
    return eval_iterations


