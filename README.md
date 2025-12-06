# USHER: Interference-Aware GPU Scheduler

Implementation of the USHER scheduler - an interference-aware GPU scheduler for DNN inference workloads.

## Installation (Local python or docker available)

### Local Installation

```bash
pip install -r requirements.txt
```

Tun the scheduler:
```bash
python3 src/main.py --gpu-type a100 --input inputs/
```

### Docker Installation

Build the Docker image:

```bash
docker build -t usher-scheduler .
```

Run the scheduler:

```bash
# Process all CSV files in inputs directory
docker run --rm -v $(pwd)/outputs:/app/outputs usher-scheduler \
  python3 src/main.py --gpu-type a100 --input inputs/
```

The `-v $(pwd)/outputs:/app/outputs` flag mounts your local `outputs` directory so results are saved to your host machine.

### Arguments

- `--cluster-type {max_goodput,min_cost}`: Scheduler mode (default: `min_cost`)
  - `max_goodput`: Maximize goodput
  - `min_cost`: Minimize cost while meeting SLO
- `--gpu-type GPU_TYPE`: GPU type for profiling data (default: `4090`)
- `--input INPUT`: Input workload CSV file (default: `input.csv`)
- `--output OUTPUT`: Output JSON file (default: `schedule_output.json`)
- `--fast`: Enable fast mode with limited search space
- `--max-configs N`: Maximum configurations to test per group

## Input Format

CSV file with header row:

```csv
Model,RPS,SLO
resnet50,1391.97,47.57630825
mobilenet_v3_large,981.59,25.97019672
```

Model names must match exactly with those in `data/<gpu_type>/profile.csv`.

## Paper Interpretation

## Assumptions

- Scheduler runs against a homogeneous GPU cluster; provide a single GPU type per invocation.
- Reported cost equals the number of GPUs provisioned (normalized cost=1 per GPU) because all GPUs share the same type.

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
```
interference_factor = c + α₁·l2_util₁ + α₂·l2_util₂ + β₁·dram_util₁ + β₂·dram_util₂
actual_latency = base_latency × interference_factor
```
