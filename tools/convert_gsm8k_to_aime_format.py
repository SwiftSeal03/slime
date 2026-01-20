#!/usr/bin/env python3
"""
Convert GSM8K parquet files to AIME-2024 format (jsonl).
AIME-2024 format: {"prompt": [{"role": "user", "content": "..."}], "label": "..."}
"""

import json
import sys
import os

def read_parquet_simple(filepath):
    """Read parquet file using available libraries."""
    try:
        import pandas as pd
        return pd.read_parquet(filepath)
    except ImportError:
        try:
            import pyarrow.parquet as pq
            return pq.read_table(filepath).to_pandas()
        except ImportError:
            try:
                from datasets import load_dataset
                ds = load_dataset('parquet', data_files=filepath, split='train')
                return ds.to_pandas()
            except ImportError:
                print("Error: Need pandas, pyarrow, or datasets library to read parquet files")
                print("Install with: pip install pandas pyarrow")
                sys.exit(1)

def convert_to_aime_format(df, output_file):
    """Convert dataframe to AIME-2024 jsonl format."""
    import ast
    import numpy as np
    with open(output_file, 'w') as f:
        for idx, row in df.iterrows():
            # The GSM8K parquet files have:
            # - 'prompt': already a list of dicts in the right format
            # - 'reward_model': dict with 'ground_truth' key containing the answer
            
            # Get the prompt - it's already in the right format
            prompt = row['prompt']
            # Convert numpy array to list if needed
            if isinstance(prompt, np.ndarray):
                prompt = prompt.tolist()
            elif isinstance(prompt, str):
                # If it's a string representation, parse it
                try:
                    prompt = ast.literal_eval(prompt)
                except:
                    # Fallback: treat as JSON string
                    prompt = json.loads(prompt)
            
            # Get the label from reward_model
            reward_model = row['reward_model']
            # Convert numpy array to dict if needed
            if isinstance(reward_model, np.ndarray):
                reward_model = reward_model.tolist()
                if isinstance(reward_model, list) and len(reward_model) > 0:
                    reward_model = reward_model[0]
            elif isinstance(reward_model, str):
                try:
                    reward_model = ast.literal_eval(reward_model)
                except:
                    reward_model = json.loads(reward_model)
            
            # Extract ground_truth
            if isinstance(reward_model, dict):
                label = reward_model.get('ground_truth', '')
            else:
                label = str(reward_model)
            
            # Format as AIME-2024 structure
            entry = {
                "prompt": prompt,
                "label": str(label)
            }
            
            f.write(json.dumps(entry) + '\n')

def main():
    nsc_data = os.environ.get('NSC', '')
    if not nsc_data:
        print("Error: NSC environment variable not set")
        sys.exit(1)
    
    gsm8k_dir = os.path.join(nsc_data, 'data', 'gsm8k')
    aime_dir = os.path.join(nsc_data, 'data', 'aime-2024')
    
    # Read the parquet files
    print("Reading train.parquet...")
    train_df = read_parquet_simple(os.path.join(gsm8k_dir, 'train.parquet'))
    print(f"Train shape: {train_df.shape}")
    print(f"Train columns: {train_df.columns.tolist()}")
    print(f"First row sample:\n{train_df.iloc[0]}\n")
    
    print("Reading test.parquet...")
    test_df = read_parquet_simple(os.path.join(gsm8k_dir, 'test.parquet'))
    print(f"Test shape: {test_df.shape}")
    print(f"Test columns: {test_df.columns.tolist()}")
    print(f"First row sample:\n{test_df.iloc[0]}\n")
    
    # Convert to AIME format
    print("Converting train.parquet to train.jsonl...")
    convert_to_aime_format(train_df, os.path.join(gsm8k_dir, 'train.jsonl'))
    
    print("Converting test.parquet to test.jsonl...")
    convert_to_aime_format(test_df, os.path.join(gsm8k_dir, 'test.jsonl'))
    
    print("Done! Created train.jsonl and test.jsonl in", gsm8k_dir)

if __name__ == '__main__':
    main()
