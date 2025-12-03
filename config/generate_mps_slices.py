#!/usr/bin/env python3
"""
Script to generate MPS slices from output JSON files.
Format: Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
"""

import json
import csv
import os
from pathlib import Path
from typing import Dict, Tuple


def load_model_info(csv_path: str) -> Dict[str, Tuple[int, str]]:
    """Load model info from CSV and return dict mapping model name to (NUM_REQS, RPS)."""
    model_info = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_name = row['Model']
            num_reqs = int(row['NUM_REQS'])
            rps = row['RPS']  # Keep as string to preserve exact precision
            model_info[model_name] = (num_reqs, rps)
    return model_info


def normalize_model_name(model_name: str) -> str:
    """Convert model name from JSON format (with hyphens) to CSV format (with underscores)."""
    return model_name.replace('-', '_')


def get_load_params(filename: str) -> Tuple[str, str, str]:
    """Get LoadStart-LoadEnd-LoadSteps based on filename."""
    if '25_to_50' in filename:
        return ('0.25', '0.25', '0.25')
    elif '50_to_75' in filename:
        return ('0.5', '0.5', '0.25')
    elif '75_to_125' in filename:
        return ('0.75', '1.25', '0.25')
    else:
        raise ValueError(f"Unknown load pattern in filename: {filename}")


def get_load_label(filename: str) -> str:
    """Get human-readable load label based on filename."""
    if '25_to_50' in filename:
        return '25% Job Mix Plan'
    elif '50_to_75' in filename:
        return '50% Job Mix Plan'
    elif '75_to_125' in filename:
        return '75-125% Job Mix Plan'
    else:
        return 'Unknown Load'


def process_output_file(json_path: str, model_info: Dict[str, Tuple[int, str]]) -> Dict[int, Dict[str, list]]:
    """Process a single output JSON file and return dict mapping GPU ID to dict of load_label -> list of slices."""
    filename = os.path.basename(json_path)
    load_start, load_end, load_steps = get_load_params(filename)
    load_label = get_load_label(filename)
    
    gpu_slices = {}
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    for gpu in data.get('gpus', []):
        gpu_id = gpu.get('id', 0)
        if gpu_id not in gpu_slices:
            gpu_slices[gpu_id] = {}
        gpu_slices[gpu_id][load_label] = []
        
        for model_data in gpu.get('models', []):
            model_name = model_data['model']
            batch_size = model_data['batch_size']
            compute_percent = round(model_data['compute_percent'])
            
            # Normalize model name to match CSV format
            normalized_name = normalize_model_name(model_name)
            
            if normalized_name in model_info:
                num_reqs, rps = model_info[normalized_name]
                
                # Format: Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
                slice_str = f"{model_name}-{batch_size}-{compute_percent}-{num_reqs}-{load_start}-{load_end}-{load_steps}-{rps}"
                gpu_slices[gpu_id][load_label].append(slice_str)
            else:
                print(f"Warning: Model '{normalized_name}' not found in model_info.csv")
    
    return gpu_slices


def get_load_order(filename: str) -> int:
    """Get order for sorting files by load range."""
    if '25_to_50' in filename:
        return 1
    elif '50_to_75' in filename:
        return 2
    elif '75_to_125' in filename:
        return 3
    else:
        return 999


def get_load_label_order(load_label: str) -> int:
    """Get order for sorting load labels."""
    if load_label.startswith('25%'):
        return 1
    elif load_label.startswith('50%'):
        return 2
    elif load_label.startswith('75-125%'):
        return 3
    else:
        return 999


def main():
    """Main function to process all output files and generate MPS slices."""
    # Get paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    outputs_dir = project_root / 'outputs'
    model_info_csv = script_dir / '_model_info.csv'
    
    # Load model info
    model_info = load_model_info(str(model_info_csv))
    
    # Process all JSON files in outputs directory, sorted by load range
    json_files = sorted(outputs_dir.glob('*.json'), key=lambda x: get_load_order(x.name))
    
    # Structure: {gpu_id: {load_label: [slices]}}
    all_gpu_slices = {}
    
    for json_file in json_files:
        print(f"Processing {json_file.name}...")
        gpu_slices = process_output_file(str(json_file), model_info)
        
        # Merge slices organized by GPU and load label
        for gpu_id, load_dict in gpu_slices.items():
            if gpu_id not in all_gpu_slices:
                all_gpu_slices[gpu_id] = {}
            for load_label, slices in load_dict.items():
                if load_label not in all_gpu_slices[gpu_id]:
                    all_gpu_slices[gpu_id][load_label] = []
                all_gpu_slices[gpu_id][load_label].extend(slices)
    
    # Print all slices grouped by GPU, then by load range
    print("\n" + "="*80)
    print("MPS Slices:")
    print("="*80)
    
    total_slices = 0
    for gpu_id in sorted(all_gpu_slices.keys()):
        print(f"\nGPU {gpu_id}:")
        gpu_data = all_gpu_slices[gpu_id]
        
        # Sort load labels by order
        sorted_load_labels = sorted(gpu_data.keys(), key=get_load_label_order)
        
        for load_label in sorted_load_labels:
            slices = gpu_data[load_label]
            if slices:  # Only print if there are slices
                print(f"  {load_label}:")
                for slice_str in slices:
                    print(f"    {slice_str}")
                    total_slices += 1
    
    print(f"\nTotal slices: {total_slices}")


if __name__ == '__main__':
    main()

