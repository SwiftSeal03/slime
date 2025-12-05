#!/usr/bin/env python3
"""
Modify arguments in run-qwen3-d6B.sh script.

Usage:
    python modify_run_script.py --n-samples-per-prompt 16 --global-batch-size 64
    python modify_run_script.py --input scripts/run-qwen3-d6B.sh --output scripts/run-qwen3-d6B-modified.sh --n-samples-per-prompt 16
    
This will read the shell script, replace the specified arguments, and write to the output file.
"""

import re
import argparse
import sys
from pathlib import Path


def parse_known_args_flexibly(argv=None):
    """
    Parse command line arguments flexibly, allowing any --arg value pairs.
    Returns (known_args, replacements_dict)
    """
    parser = argparse.ArgumentParser(
        description="Modify arguments in a shell script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python modify_run_script.py --n-samples-per-prompt 16
    python modify_run_script.py --input scripts/run-qwen3-d6B.sh --global-batch-size 64
    python modify_run_script.py -o modified.sh --tensor-model-parallel-size 4
        """
    )
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='Input shell script path (default: ../scripts/run-qwen3-d6B.sh)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output shell script path (default: overwrites input)')
    parser.add_argument('--dry-run', '-n', action='store_true',
                        help='Print changes without writing to file')
    
    # Parse known args first
    known_args, remaining = parser.parse_known_args(argv)
    
    # Parse remaining args as key-value pairs for replacement
    replacements = {}
    i = 0
    while i < len(remaining):
        arg = remaining[i]
        if arg.startswith('--'):
            key = arg
            # Check if next arg is a value (not another --arg)
            if i + 1 < len(remaining) and not remaining[i + 1].startswith('--'):
                value = remaining[i + 1]
                i += 2
            else:
                # Boolean flag or missing value
                value = None
                i += 1
            replacements[key] = value
        else:
            print(f"Warning: Unexpected argument '{arg}', skipping", file=sys.stderr)
            i += 1
    
    return known_args, replacements


def replace_arg_in_script(content: str, arg_name: str, new_value: str) -> tuple[str, bool]:
    """
    Replace an argument value in the shell script content.
    
    Handles formats like:
        --arg value
        --arg=value
    
    Returns (modified_content, was_replaced)
    """
    # Escape special regex characters in arg_name
    escaped_arg = re.escape(arg_name)
    
    # Pattern 1: --arg value (space separated, value on same line)
    pattern1 = rf'({escaped_arg})\s+([^\s\\#\n]+)'
    
    # Pattern 2: --arg=value (equals separated)
    pattern2 = rf'({escaped_arg})=([^\s\\#\n]+)'
    
    replaced = False
    
    # Try pattern 1 first (space separated)
    def replace_func1(match):
        nonlocal replaced
        replaced = True
        return f'{match.group(1)} {new_value}'
    
    new_content = re.sub(pattern1, replace_func1, content)
    
    if not replaced:
        # Try pattern 2 (equals separated)
        def replace_func2(match):
            nonlocal replaced
            replaced = True
            return f'{match.group(1)}={new_value}'
        
        new_content = re.sub(pattern2, replace_func2, content)
    
    return new_content, replaced


def main():
    args, replacements = parse_known_args_flexibly()
    
    if not replacements:
        print("No replacements specified. Use --arg value pairs to specify replacements.")
        print("Example: python modify_run_script.py --n-samples-per-prompt 16")
        sys.exit(1)
    
    # Determine input path
    script_dir = Path(__file__).parent.resolve()
    if args.input:
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = script_dir / args.input
    else:
        input_path = script_dir.parent / 'scripts' / 'run-qwen3-d6B.sh'
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = script_dir / args.output
    else:
        output_path = input_path  # Overwrite input by default
    
    # Read input file
    content = input_path.read_text()
    original_content = content
    
    # Apply replacements
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"\nReplacements:")
    
    successful = []
    failed = []
    
    for arg_name, new_value in replacements.items():
        if new_value is None:
            print(f"  Warning: {arg_name} has no value, skipping")
            continue
        
        content, was_replaced = replace_arg_in_script(content, arg_name, new_value)
        
        if was_replaced:
            print(f"  ✓ {arg_name} -> {new_value}")
            successful.append(arg_name)
        else:
            print(f"  ✗ {arg_name} not found in script")
            failed.append(arg_name)
    
    # Show summary
    print(f"\nSummary: {len(successful)} replaced, {len(failed)} not found")
    
    if failed:
        print(f"Not found: {', '.join(failed)}")
    
    # Write output
    if args.dry_run:
        print("\n[Dry run - no changes written]")
        if content != original_content:
            print("\nChanges would be:")
            # Show diff-like output
            orig_lines = original_content.splitlines()
            new_lines = content.splitlines()
            for i, (orig, new) in enumerate(zip(orig_lines, new_lines), 1):
                if orig != new:
                    print(f"  Line {i}:")
                    print(f"    - {orig.strip()}")
                    print(f"    + {new.strip()}")
    else:
        if content != original_content:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content)
            print(f"\nWritten to: {output_path}")
        else:
            print("\nNo changes made.")


if __name__ == "__main__":
    main()



