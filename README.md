# USHER: Interference-Aware GPU Scheduler

Implementation of the USHER scheduler - an interference-aware GPU scheduler that maximizes resource utilization for DNN inference workloads by intelligently multiplexing compute-heavy and memory-heavy models.

## Features

- **GK-Estimator**: Estimates compute (C_req) and memory (M_req) requirements with linear interpolation
- **Interference Modeling**: Accounts for L2 cache and DRAM contention between co-located models
- **Model Grouping**: K-means variant clustering to balance ∑C_req ≈ ∑M_req
- **Intelligent Placement**: Pairs C-heavy and M-heavy models to maximize GPU utilization
- **Fast Mode**: Optimized search for large workloads (55× faster than exhaustive)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Fast mode (recommended for most use cases)
python src/main.py --fast --cluster-type non-fixed --gpu-type 4090

# Exhaustive mode (paper-accurate, slower)
python src/main.py --cluster-type non-fixed --gpu-type 4090
```

## Usage

### Arguments

- `--cluster-type {fixed,non-fixed}`: Scheduler mode (default: `non-fixed`)
  - `fixed`: Maximize goodput with existing GPUs
  - `non-fixed`: Minimize cost by adding GPUs as needed
- `--gpu-type GPU_TYPE`: GPU type for profiling data (default: `4090`)
- `--input INPUT`: Input workload CSV file (default: `input.csv`)
- `--output OUTPUT`: Output JSON file (default: `schedule_output.json`)
- `--fast`: Enable fast mode with limited search space (recommended for large workloads)
- `--max-configs N`: Maximum configurations to test per group

### Examples

**Fast mode (recommended):**
```bash
python src/main.py --fast --cluster-type non-fixed --gpu-type 4090
```

**Exhaustive search:**
```bash
python src/main.py --cluster-type non-fixed --gpu-type 4090
```

**Custom configuration limit:**
```bash
python src/main.py --fast --max-configs 10000 --input workload.csv
```

## Input Format

`input.csv` specifies the workload:

```
<number_of_models>
<model_id>, <rps>, <slo_ms>
...
```

Example:
```
3
6, 25, 66     # densenet161, 25 RPS, 66ms SLO
7, 20, 108    # mobilenet_v2, 20 RPS, 108ms SLO
9, 15, 142    # bert, 15 RPS, 142ms SLO
```

Model IDs map to names in `config/mem-config.json`:
1. lenet
2. googlenet
3. resnet50
4. ssd-mobilenetv1
5. vgg16
6. densenet161
7. mobilenet_v2
8. mnasnet1_0
9. bert

## Output

### Console Output
- Interference modeling status
- Workload requests
- Model grouping results
- Configuration space estimation
- Per-group scheduling decisions
- Final GPU assignments with utilization and interference factors

### JSON Output

```json
{
  "cluster_type": "non-fixed",
  "gpu_type": "4090",
  "total_gpus": 1,
  "assignments": [
    {
      "model": "densenet161",
      "batch_size": 8,
      "gpu_id": 0,
      "gpu_type": "4090"
    }
  ]
}
```

## Configuration Files

### `config/device-config.json`
GPU specifications and profiling data paths.

### `config/mem-config.json`
Model-to-ID mappings and memory requirements.

### `config/sched-config.json`
Scheduler parameters.

## Algorithm Overview

### 1. Model Grouping
- Calculate average C_req and M_req for each model
- Iteratively merge groups to minimize D = |∑C_req - ∑M_req|
- Creates balanced groups of up to 4 models

### 2. Scheduling
For each group:
- Compute cl_m (minimum replication degree) based on SLO
- Generate configurations: BS ∈ {4, 8, 16, 32, 64, 128}, RD ∈ {cl_m, 2·cl_m, ..., 6·cl_m}
- Test each configuration with placement algorithm
- Select based on cluster type (minimize cost or maximize goodput)

### 3. Placement
- Classify models as C-heavy (C_req/M_req ≥ 1.2) or M-heavy (M_req/C_req ≥ 1.2)
- Sort by C_req + M_req, pair C-heavy with M-heavy
- Place replicas on existing GPUs (prioritize least fragmentation)
- Initialize new GPUs only when necessary

### 4. Interference Modeling
Uses constants from `int_model_constant.csv`:
```
interference_factor = c + α₁·l2_util₁ + α₂·l2_util₂ + β₁·dram_util₁ + β₂·dram_util₂
actual_latency = base_latency × interference_factor
```

## Performance: Fast vs Exhaustive Mode

**3 Models:**
- Exhaustive: ~3.3 seconds (46,656 configs)
- Fast: ~0.06 seconds (5,000 configs)
- **Speedup: 55×**

**50 Models (13 groups of 4):**
- Exhaustive: ~5.6 hours
- Fast: ~6 minutes
- **Speedup: 56×**

Fast mode limits:
- BS options: [8, 16, 32, 64] (skip extremes)
- RD multipliers: up to 3× instead of 6×
- Max 5,000 configs per group
- Early termination when optimal found

## Project Structure

```
Usher/
├── src/
│   ├── main.py              # Scheduler, grouping, placement
│   ├── gk_estimator.py      # C_req/M_req estimation, interference
│   ├── objects_dataclass.py # Data structures
│   └── file_parse.py        # Input/config parsing
├── config/
│   ├── device-config.json   # GPU specifications
│   ├── mem-config.json      # Model mappings
│   └── sched-config.json    # Scheduler config
├── data/
│   └── 4090/
│       ├── profile.csv           # Model profiling data
│       ├── latency.csv           # Latency measurements
│       └── int_model_constant.csv # Interference constants
├── input.csv               # Workload specification
└── requirements.txt        # Dependencies (numpy)
```

## Adding New GPU Types

To add A100 or other GPUs:

1. Add profiling data to `data/<gpu_type>/`:
   - `profile.csv` (SM util, memory util, L2 util, latency)
   - `latency.csv` (partition-specific latencies)
   - `int_model_constant.csv` (interference constants)

2. Update `config/device-config.json`:
   ```json
   {
     "device_specs": [{
       "type": "a100",
       "mem_mb": 81920,
       "latency_prof_file": "data/a100/latency.csv",
       "interference_const_file": "data/a100/int_model_constant.csv",
       "interference_util_file": "data/a100/profile.csv"
     }]
   }
   ```

3. Run with:
   ```bash
   python src/main.py --gpu-type a100 --fast
   ```

## Implementation Notes

- **C_req**: Normalized SM utilization [0, 1]
- **M_req**: Normalized memory utilization [0, 1]  
- **Interpolation**: Linear (numpy.interp) for missing batch sizes
- **Grouping**: Max 4 models per group (2^2)
- **Interference**: Enabled automatically if `int_model_constant.csv` exists

## Recommendations

- **Development/Testing**: Use `--fast` mode
- **Large workloads (>10 models)**: Always use `--fast`
- **Paper comparisons**: Use exhaustive mode (no --fast flag)
- **Custom tuning**: Use `--max-configs` to balance speed vs accuracy

## Baseline Comparison

This implementation serves as a baseline for comparing against other schedulers (GPUlets, etc.).