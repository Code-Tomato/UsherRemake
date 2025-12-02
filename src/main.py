from file_parse import parse_input_workload, load_gpu_config
from gk_estimator import GKEstimator, compute_cl_m
from objects_dataclass import WorkloadRequest, GPU, Assignment, Classification
import json
import argparse
import sys
from copy import deepcopy

def group_models(workloads, estimator, gpu_specs, max_models_per_group=4):
   """
   Group models using k-means variant where sum(C_req) ≈ sum(M_req).
   Distance D = |sum(C_req) - sum(M_req)| is minimized within groups.
   """
   # Get max GPU memory from specs
   max_gpu_mem = max(spec['mem_mb'] for spec in gpu_specs)
   
   # Calculate average C_req and M_req for each workload
   workload_reqs = []
   for wl in workloads:
      c_req, m_req = estimator.compute_average_c_req_m_req(wl.model_name, max_gpu_mem)
      workload_reqs.append({
         'workload': wl,
         'c_req': c_req,
         'm_req': m_req
      })
   
   # Start with each model in its own group
   groups = [[wr] for wr in workload_reqs]
   
   # Determine number of passes (p) such that 2^p = max_models_per_group
   import math
   num_passes = int(math.log2(max_models_per_group))
   
   # Perform k-means-like merging
   for pass_num in range(num_passes):
      if len(groups) <= 1:
         break
      
      # Calculate distance for each group
      def group_distance(group):
         sum_c = sum(item['c_req'] for item in group)
         sum_m = sum(item['m_req'] for item in group)
         return abs(sum_c - sum_m)
      
      # Find best pairs to merge (minimize distance after merging)
      new_groups = []
      used = set()
      
      while len(used) < len(groups):
         best_pair = None
         best_distance = float('inf')
         
         for i in range(len(groups)):
            if i in used:
               continue
            for j in range(i + 1, len(groups)):
               if j in used:
                  continue
               
               merged = groups[i] + groups[j]
               dist = group_distance(merged)
               
               if dist < best_distance:
                  best_distance = dist
                  best_pair = (i, j)
         
         if best_pair is None:
            # Only one group left
            for i in range(len(groups)):
               if i not in used:
                  new_groups.append(groups[i])
                  used.add(i)
            break
         
         # Merge the best pair
         i, j = best_pair
         new_groups.append(groups[i] + groups[j])
         used.add(i)
         used.add(j)
      
      groups = new_groups
   
   # Convert back to list of workloads
   result_groups = []
   for group in groups:
      result_groups.append([item['workload'] for item in group])
   
   return result_groups

def placement(group_workloads, configurations, gpu_pool, estimator, cluster_type, gpu_specs):
   """
   Placement algorithm (Algorithm 2 from paper).
   Returns: (cost, total_goodput, gpu_pool_updated, assignments)
   """
   gpu_pool = deepcopy(gpu_pool)
   assignments = []
   
   # Create a mapping of workload -> (BS, RD)
   config_map = {}
   for i, wl in enumerate(group_workloads):
      config_map[wl.model_name] = configurations[i]
   
   # Classify models as C-heavy, M-heavy, or balanced
   model_info = []
   for wl in group_workloads:
      bs, rd = config_map[wl.model_name]
      c_req, m_req = estimator.get_c_req_m_req(wl.model_name, bs)
      
      ratio = c_req / m_req if m_req > 0.001 else 1.0
      if ratio >= 1.2:
         classify = Classification.C_HEAVY
      elif ratio <= 0.8:
         classify = Classification.M_HEAVY
      else:
         classify = Classification.BALANCED
      
      model_info.append({
         'workload': wl,
         'bs': bs,
         'rd': rd,
         'c_req': c_req,
         'm_req': m_req,
         'classify': classify
      })
   
   # Separate into C-heavy and M-heavy
   c_heavy = [m for m in model_info if m['classify'] == Classification.C_HEAVY]
   m_heavy = [m for m in model_info if m['classify'] == Classification.M_HEAVY]
   balanced = [m for m in model_info if m['classify'] == Classification.BALANCED]
   
   # Sort by descending C_req + M_req
   c_heavy.sort(key=lambda x: x['c_req'] + x['m_req'], reverse=True)
   m_heavy.sort(key=lambda x: x['c_req'] + x['m_req'], reverse=True)
   balanced.sort(key=lambda x: x['c_req'] + x['m_req'], reverse=True)
   
   # Pair C-heavy with M-heavy
   final_model_list = []
   for i in range(max(len(c_heavy), len(m_heavy))):
      if i < len(c_heavy):
         final_model_list.append(c_heavy[i])
      if i < len(m_heavy):
         final_model_list.append(m_heavy[i])
   
   # Add balanced models
   final_model_list.extend(balanced)
   
   # Place each model replica
   total_cost = 0
   for model_info_item in final_model_list:
      wl = model_info_item['workload']
      bs = model_info_item['bs']
      rd = model_info_item['rd']
      c_req = model_info_item['c_req']
      m_req = model_info_item['m_req']
      
      # Place 'rd' replicas of this model
      for replica_id in range(rd):
         placed = False
         
         # Try to place in existing GPUs (prioritize those with least remaining space)
         gpu_candidates = [(gpu, gpu.remaining_space()) for gpu in gpu_pool if gpu.can_fit(c_req, m_req)]
         gpu_candidates.sort(key=lambda x: x[1])  # Ascending remaining space
         
         for gpu, _ in gpu_candidates:
            if gpu.can_fit(c_req, m_req):
               gpu.compute_used += c_req
               gpu.memory_used += m_req * gpu.max_memory  # Convert fraction to absolute memory
               gpu.model_replicas.append((wl.model_name, bs, replica_id))
               assignments.append(Assignment(wl.model_name, bs, replica_id, gpu.gpu_id, gpu.gpu_type))
               placed = True
               break
         
         # If not placed, create a new GPU
         if not placed:
            # Find cheapest GPU type that can fit
            best_gpu_spec = None
            for spec in gpu_specs:
               max_c = 1.0  # Assume 100% compute
               max_m = spec['mem_mb']
               if c_req <= max_c and m_req * max_m <= max_m:  # m_req is a fraction
                  if best_gpu_spec is None or spec['mem_mb'] < best_gpu_spec['mem_mb']:
                     best_gpu_spec = spec
            
            if best_gpu_spec is None:
               # Can't fit anywhere - configuration is infeasible
               return float('inf'), 0, gpu_pool, []
            
            new_gpu_id = len(gpu_pool)
            new_gpu = GPU(
               gpu_id=new_gpu_id,
               gpu_type=best_gpu_spec['type'],
               max_compute=1.0,
               max_memory=best_gpu_spec['mem_mb'],
               compute_used=c_req,
               memory_used=m_req * best_gpu_spec['mem_mb'],  # Scale m_req to actual memory
               cost=1.0
            )
            new_gpu.model_replicas.append((wl.model_name, bs, replica_id))
            gpu_pool.append(new_gpu)
            assignments.append(Assignment(wl.model_name, bs, replica_id, new_gpu.gpu_id, new_gpu.gpu_type))
            total_cost += new_gpu.cost
   
   # Calculate total goodput with interference modeling
   total_goodput = 0
   for model_info_item in final_model_list:
      wl = model_info_item['workload']
      bs = model_info_item['bs']
      rd = model_info_item['rd']
      
      # Get base latency
      base_latency_ms = estimator.get_latency(wl.model_name, bs)
      
      # Calculate interference-aware latency for each replica
      # For each replica, find which GPU it's on and calculate interference
      total_throughput = 0
      for replica_id in range(rd):
         # Find which GPU this replica is on
         replica_gpu = None
         for gpu in gpu_pool:
            for model_name, replica_bs, r_id in gpu.model_replicas:
               if model_name == wl.model_name and replica_bs == bs and r_id == replica_id:
                  replica_gpu = gpu
                  break
            if replica_gpu:
               break
         
         if replica_gpu:
            # Get all models on this GPU for interference calculation
            models_on_gpu = [(m_name, m_bs) for m_name, m_bs, _ in replica_gpu.model_replicas]
            interference_factor = estimator.calculate_interference_factor(models_on_gpu)
            
            # Apply interference to latency
            actual_latency_ms = base_latency_ms * interference_factor
         else:
            actual_latency_ms = base_latency_ms
         
         # Calculate throughput for this replica
         throughput_per_replica = (bs / actual_latency_ms) * 1000  # requests/sec
         total_throughput += throughput_per_replica
      
      # Goodput is min of achieved and required
      goodput = min(total_throughput, wl.rps)
      total_goodput += goodput
   
   return total_cost, total_goodput, gpu_pool, assignments

def scheduler(workload_groups, estimator, cluster_type, gpu_specs, initial_gpus=None, fast_mode=False, max_configs_per_group=None):
   """
   Scheduler algorithm (Algorithm 1 from paper).
   
   Args:
      fast_mode: If True, use heuristics to limit search space
      max_configs_per_group: Maximum configurations to test per group
   """
   all_assignments = []
   gpu_pool = initial_gpus if initial_gpus else []
   
   for group_workloads in workload_groups:
      print(f"\n{'='*60}")
      print(f"Scheduling group with {len(group_workloads)} models:")
      for wl in group_workloads:
         print(f"  - {wl}")
      
      # Calculate cl_m for each model
      cl_m_map = {}
      for wl in group_workloads:
         max_gpu_mem = max(spec['mem_mb'] for spec in gpu_specs)
         cl_m = compute_cl_m(wl, wl.model_name, wl.slo_ms, max_gpu_mem, estimator)
         cl_m_map[wl.model_name] = cl_m
         print(f"  cl_m for {wl.model_name}: {cl_m}")
      
      # Get available batch sizes for each model
      bs_options_per_model = []
      for wl in group_workloads:
         available_bs = estimator.get_available_batch_sizes(wl.model_name)
         bs_options_per_model.append(available_bs)
         print(f"  {wl.model_name} available batch sizes: {sorted(available_bs)}")
      
      # Configure search space based on mode
      if fast_mode:
         max_multiplier = 3  # Only try up to 3x replication
         if max_configs_per_group is None:
            max_configs_per_group = 5000  # Limit to 5K configs
         print(f"  Fast mode: Testing up to {max_configs_per_group:,} configurations")
      else:
         max_multiplier = 6  # Paper says m = 1, 2, ..., 6
         print(f"  Exhaustive mode: Testing all configurations")
      
      # Calculate total combinations for estimation
      bs_combos = 1
      for bs_list in bs_options_per_model:
         bs_combos *= len(bs_list)
      rd_combos = max_multiplier ** len(group_workloads)
      total_configs = bs_combos * rd_combos
      estimated_time = total_configs * 0.000036  # Measured empirically (~0.036 ms per config)
      print(f"  Estimated configurations: {bs_combos:,} BS combos × {rd_combos:,} RD combos = {total_configs:,} total")
      if estimated_time < 60:
         print(f"  Estimated time: {estimated_time:.1f} seconds")
      elif estimated_time < 3600:
         print(f"  Estimated time: {estimated_time/60:.1f} minutes")
      else:
         print(f"  Estimated time: {estimated_time/3600:.1f} hours")
      
      best_config = None
      best_cost = float('inf')
      best_goodput = 0
      best_gpu_pool = None
      best_assignments = []
      configs_tested = 0
      
      for bs_combo in generate_bs_combinations(group_workloads, estimator, fast_mode):
         for rd_combo in generate_rd_combinations(group_workloads, cl_m_map, max_multiplier=max_multiplier):
            # Check if we've hit the config limit
            if max_configs_per_group and configs_tested >= max_configs_per_group:
               print(f"  Reached configuration limit ({max_configs_per_group:,}), stopping search")
               break
            configurations = list(zip(bs_combo, rd_combo))
            configs_tested += 1
            
            cost, goodput, updated_gpu_pool, assignments = placement(
               group_workloads, configurations, gpu_pool, estimator, cluster_type, gpu_specs
            )
            
            # Select based on cluster type
            if cluster_type == 'max_goodput':
               # Maximize goodput
               if goodput > best_goodput:
                  best_goodput = goodput
                  best_cost = cost
                  best_config = configurations
                  best_gpu_pool = updated_gpu_pool
                  best_assignments = assignments
            else:  # min_cost
               # Minimize cost while meeting SLO
               total_workload = sum(wl.rps for wl in group_workloads)
               if goodput >= total_workload and cost < best_cost:
                  best_cost = cost
                  best_goodput = goodput
                  best_config = configurations
                  best_gpu_pool = updated_gpu_pool
                  best_assignments = assignments
         
         # Break outer loop if we hit config limit
         if max_configs_per_group and configs_tested >= max_configs_per_group:
            break
      
      # Update GPU pool and add assignments
      if best_config:
         gpu_pool = best_gpu_pool
         all_assignments.extend(best_assignments)
         print(f"\nBest configuration for group (tested {configs_tested:,} configs):")
         for i, wl in enumerate(group_workloads):
            bs, rd = best_config[i]
            print(f"  {wl.model_name}: BS={bs}, RD={rd}")
         print(f"  Cost: {best_cost:.2f}, Goodput: {best_goodput:.2f}")
      else:
         print(f"  WARNING: No feasible configuration found for group!")
   
   return gpu_pool, all_assignments

def generate_bs_combinations(workloads, estimator, fast_mode=False):
   """
   Generate batch size combinations using only available batch sizes for each model.
   """
   import itertools
   
   # Get available batch sizes for each model
   bs_options_per_model = []
   for wl in workloads:
      available_bs = estimator.get_available_batch_sizes(wl.model_name)
      if not available_bs:
         # Fallback if no data
         available_bs = [1, 4, 8, 16, 32] if not fast_mode else [8, 16, 32]
      
      # Filter to reasonable range based on mode
      if fast_mode:
         # Fast mode: only use common batch sizes
         filtered_bs = [bs for bs in available_bs if bs in [8, 16, 32, 64]]
         if not filtered_bs:
            filtered_bs = available_bs[:3] if len(available_bs) >= 3 else available_bs
         bs_options_per_model.append(filtered_bs)
      else:
         # Exhaustive mode: use all available batch sizes
         bs_options_per_model.append(available_bs)
   
   # Generate all combinations
   all_combos = list(itertools.product(*bs_options_per_model))
   
   return all_combos

def generate_rd_combinations(workloads, cl_m_map, max_multiplier=6):
   """
   Generate replication degree combinations.
   Paper specifies: RD ∈ {m · cl_M}, m = 1, 2, ..., 6
   """
   import itertools
   
   rd_options_per_model = []
   for wl in workloads:
      cl_m = cl_m_map[wl.model_name]
      # Paper: RD = m * cl_m where m = 1, 2, ..., max_multiplier
      options = [cl_m * m for m in range(1, max_multiplier + 1)]
      rd_options_per_model.append(options)
   
   return list(itertools.product(*rd_options_per_model))

def main():
   parser = argparse.ArgumentParser(description='USHER Scheduler')
   parser.add_argument('--cluster-type', choices=['max_goodput', 'min_cost'], default='min_cost',
                      help='Cluster type: max_goodput (maximize goodput) or min_cost (minimize cost)')
   parser.add_argument('--gpu-type', default='4090', help='GPU type to use')
   parser.add_argument('--input', default='input.csv', help='Input workload file')
   parser.add_argument('--output', default='schedule_output.json', help='Output JSON file')
   parser.add_argument('--fast', action='store_true',
                      help='Fast mode: limit search space for speed (recommended for large workloads)')
   parser.add_argument('--max-configs', type=int, default=None,
                      help='Maximum configurations to test per group (default: unlimited in normal mode, 5000 in fast mode)')
   
   args = parser.parse_args()
   
   print(f"USHER Scheduler - Cluster Type: {args.cluster_type}, GPU: {args.gpu_type}")
   print("="*60)
   
   # Load configurations
   all_gpu_specs = load_gpu_config()
   
   # Filter to only the requested GPU type
   gpu_specs = [spec for spec in all_gpu_specs if spec['type'] == args.gpu_type]
   if not gpu_specs:
      raise ValueError(f"GPU type '{args.gpu_type}' not found in device-config.json. Available types: {[s['type'] for s in all_gpu_specs]}")
   
   # Initialize GK-Estimator
   estimator = GKEstimator(gpu_type=args.gpu_type)
   
   # Show interference modeling status
   if estimator.interference_constants.get('constant', 1.0) != 1.0:
      print(f"Interference modeling: ENABLED")
      print(f"  Constants: α=[{estimator.interference_constants['l2_util_coef1']:.3f}, "
            f"{estimator.interference_constants['l2_util_coef2']:.3f}], "
            f"β=[{estimator.interference_constants['dram_util_coef1']:.3f}, "
            f"{estimator.interference_constants['dram_util_coef2']:.3f}], "
            f"c={estimator.interference_constants['constant']:.3f}")
   else:
      print(f"Interference modeling: DISABLED (using base latencies)")
   
   # Parse input workload
   workloads = parse_input_workload(args.input, args.gpu_type)
   print(f"\nLoaded {len(workloads)} workload requests:")
   for wl in workloads:
      print(f"  {wl}")
   
   # Group models
   print(f"\n{'='*60}")
   print("Grouping models...")
   workload_groups = group_models(workloads, estimator, gpu_specs)
   print(f"Created {len(workload_groups)} groups")
   
   # Estimate configuration space and runtime
   print(f"\n{'='*60}")
   print("Configuration Space Estimation:")
   
   for i, group in enumerate(workload_groups):
      num_models = len(group)
      
      # Get actual available batch sizes for each model
      bs_options_per_model = []
      for wl in group:
         available_bs = estimator.get_available_batch_sizes(wl.model_name)
         if not available_bs:
            available_bs = [1, 4, 8, 16, 32]  # Fallback
         bs_options_per_model.append(available_bs)
      
      # Calculate actual combinations
      bs_combos = 1
      for bs_list in bs_options_per_model:
         bs_combos *= len(bs_list)
      
      rd_combos = 6 ** num_models  # RD multipliers: 1-6x
      total_configs = bs_combos * rd_combos
      
      # Estimate time: ~0.000036 seconds per configuration (measured empirically)
      estimated_seconds = total_configs * 0.000036
      
      print(f"\nGroup {i+1} ({num_models} models):")
      bs_counts = [len(bs_list) for bs_list in bs_options_per_model]
      bs_formula = " × ".join(str(c) for c in bs_counts)
      print(f"  BS combinations: {bs_formula} = {bs_combos:,}")
      print(f"  RD combinations: 6^{num_models} = {rd_combos:,}")
      print(f"  Total configurations: {total_configs:,}")
      
      if estimated_seconds < 60:
         print(f"  Estimated time: {estimated_seconds:.1f} seconds")
      elif estimated_seconds < 3600:
         print(f"  Estimated time: {estimated_seconds/60:.1f} minutes")
      elif estimated_seconds < 86400:
         print(f"  Estimated time: {estimated_seconds/3600:.1f} hours")
      else:
         print(f"  Estimated time: {estimated_seconds/86400:.1f} days")
      
      if total_configs > 100000000:  # 100 million
         print(f"  WARNING: Configuration space is extremely large!")
         print(f"           Consider using --fast mode or --max-configs to limit search space.")
   
   print(f"\n{'='*60}")
   if args.fast:
      print("Starting scheduler in FAST MODE...")
      print("(Limited search space for speed)")
   else:
      print("Starting scheduler in EXHAUSTIVE MODE...")
      print("(Testing all configurations - may take a long time)")
   
   import time
   start_time = time.time()
   
   # Schedule
   gpu_pool, assignments = scheduler(workload_groups, estimator, args.cluster_type, gpu_specs, 
                                     fast_mode=args.fast, max_configs_per_group=args.max_configs)
   
   end_time = time.time()
   elapsed = end_time - start_time
   print(f"\nScheduling completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
   
   # Print results
   print(f"\n{'='*60}")
   print("FINAL SCHEDULE:")
   print(f"{'='*60}")
   print(f"Total GPUs used: {len(gpu_pool)}")
   for gpu in gpu_pool:
      print(f"\n{gpu}")
      
      # Calculate and show interference factor if multiple models
      if len(gpu.model_replicas) > 1:
         models_on_gpu = [(m_name, m_bs) for m_name, m_bs, _ in gpu.model_replicas]
         interference_factor = estimator.calculate_interference_factor(models_on_gpu)
         print(f"    Interference factor: {interference_factor:.3f}x")
      
      for model_name, bs, replica_id in gpu.model_replicas:
         print(f"    - {model_name} (BS={bs}, Replica={replica_id})")
   
   # Export to JSON
   output_data = {
      "cluster_type": args.cluster_type,
      "gpu_type": args.gpu_type,
      "total_gpus": len(gpu_pool),
      "assignments": [a.to_dict(include_replica_id=False) for a in assignments]
   }
   
   with open(args.output, 'w') as f:
      json.dump(output_data, f, indent=2)
   
   print(f"\n{'='*60}")
   print(f"Schedule exported to {args.output}")
   print(f"{'='*60}")

if __name__ == "__main__":
   main()
