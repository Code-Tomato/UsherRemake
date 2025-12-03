import numpy as np

class GKEstimator:
   def __init__(self, gpu_type):
      self.gpu_type = gpu_type
      self.profile_data = {}  # {model_name: {batch_size: {c_util, m_util, l2_util}}}
      self.latency_data = {}  # {model_name: {partition: {batch_size: latency}}}
      self.interference_constants = {}  # Interference modeling constants
      self.load_profile_data()
      self.load_latency_data()
      self.load_interference_constants()
   
   def load_profile_data(self):
      """Load profiling data from profile.csv"""
      profile_path = f'data/{self.gpu_type}/profile.csv'
      with open(profile_path, 'r') as f:
         lines = f.readlines()
      
      for line in lines[1:]:
         parts = line.strip().split(',')
         model_name = parts[0].strip()
         batch_size = int(parts[1].strip())
         sm_util = float(parts[2].strip())
         l2_util = float(parts[3].strip())
         mem_util = float(parts[4].strip())
         
         if model_name not in self.profile_data:
            self.profile_data[model_name] = {}
         
         self.profile_data[model_name][batch_size] = {
            'c_util': sm_util,
            'm_util': mem_util,
            'l2_util': l2_util
         }
   
   def load_latency_data(self):
      """Load latency data from latency.csv"""
      latency_path = f'data/{self.gpu_type}/latency.csv'
      with open(latency_path, 'r') as f:
         lines = f.readlines()
      
      for line in lines[1:]:
         parts = line.strip().split(',')
         model_name = parts[0].strip()
         partition = int(parts[1].strip())
         batch_size = int(parts[2].strip())
         latency = float(parts[3].strip())
         
         if model_name not in self.latency_data:
            self.latency_data[model_name] = {}
         if partition not in self.latency_data[model_name]:
            self.latency_data[model_name][partition] = {}
         
         self.latency_data[model_name][partition][batch_size] = latency
   
   def load_interference_constants(self):
      """Load interference modeling constants from int_model_constant.csv"""
      const_path = f'data/{self.gpu_type}/int_model_constant.csv'
      
      with open(const_path, 'r') as f:
         lines = f.readlines()
      
      if len(lines) < 2:
         raise ValueError(f"Invalid int_model_constant.csv: expected at least 2 lines (header + data), got {len(lines)}")
      
      parts = lines[1].strip().split(',')
      if len(parts) < 5:
         raise ValueError(f"Invalid int_model_constant.csv: expected 5 values, got {len(parts)}")
      
      self.interference_constants = {
         'l2_util_coef1': float(parts[0]),
         'l2_util_coef2': float(parts[1]),
         'dram_util_coef1': float(parts[2]),
         'dram_util_coef2': float(parts[3]),
         'constant': float(parts[4])
      }
   
   def get_available_batch_sizes(self, model_name):
      """Get list of available batch sizes for a model"""
      if model_name not in self.profile_data:
         return []
      return sorted(self.profile_data[model_name].keys())
   
   def get_c_req_m_req(self, model_name, batch_size):
      """Get C_req and M_req for a model at a specific batch size"""
      if model_name not in self.profile_data:
         return 0.5, 0.5  # Default values
      
      batch_sizes = sorted(self.profile_data[model_name].keys())
      
      # If exact batch size exists, return it
      if batch_size in self.profile_data[model_name]:
         data = self.profile_data[model_name][batch_size]
         return data['c_util'] / 100.0, data['m_util'] / 100.0
      
      # Otherwise, use linear interpolation
      if len(batch_sizes) < 2:
         # Not enough data points for interpolation
         data = self.profile_data[model_name][batch_sizes[0]]
         return data['c_util'] / 100.0, data['m_util'] / 100.0
      
      c_utils = [self.profile_data[model_name][bs]['c_util'] for bs in batch_sizes]
      m_utils = [self.profile_data[model_name][bs]['m_util'] for bs in batch_sizes]
      
      # Linear interpolation using numpy
      c_req = float(np.interp(batch_size, batch_sizes, c_utils)) / 100.0
      m_req = float(np.interp(batch_size, batch_sizes, m_utils)) / 100.0
      
      # Clamp values to reasonable ranges
      c_req = max(0.01, min(1.0, c_req))
      m_req = max(0.01, min(1.0, m_req))
      
      return c_req, m_req
   
   def get_latency(self, model_name, batch_size, partition=100):
      """Get latency for a model at a specific batch size and partition"""
      if model_name not in self.latency_data or partition not in self.latency_data[model_name]:
         raise ValueError(f"Latency data not found for model '{model_name}' at partition {partition}")
      
      batch_sizes = sorted(self.latency_data[model_name][partition].keys())
      
      if batch_size in self.latency_data[model_name][partition]:
         return self.latency_data[model_name][partition][batch_size]
      
      # Linear interpolation
      if len(batch_sizes) < 2:
         return self.latency_data[model_name][partition][batch_sizes[0]]
      
      latencies = [self.latency_data[model_name][partition][bs] for bs in batch_sizes]
      
      return float(np.interp(batch_size, batch_sizes, latencies))
   
   def compute_average_c_req_m_req(self, model_name, max_memory_mb):
      """Compute average C_req and M_req across batch sizes that fit on the GPU."""
      if model_name not in self.profile_data:
         return 0.5, 0.5
      
      batch_sizes = sorted(self.profile_data[model_name].keys())
      valid_c_reqs = []
      valid_m_reqs = []
      
      for bs in batch_sizes:
         profile_entry = self.profile_data[model_name][bs]
         c_util = profile_entry['c_util'] / 100.0
         m_util = profile_entry['m_util'] / 100.0
         
         required_memory_mb = m_util * max_memory_mb
         if c_util > 1.0 or required_memory_mb > max_memory_mb:
            break
         
         valid_c_reqs.append(c_util)
         valid_m_reqs.append(m_util)
      
      if not valid_c_reqs:
         return 0.5, 0.5
      
      return float(np.mean(valid_c_reqs)), float(np.mean(valid_m_reqs))
   
   def get_l2_util(self, model_name, batch_size):
      """Get L2 cache utilization for interference modeling"""
      if model_name not in self.profile_data:
         return 0.0
      
      batch_sizes = sorted(self.profile_data[model_name].keys())
      
      if batch_size in self.profile_data[model_name]:
         return self.profile_data[model_name][batch_size]['l2_util'] / 100.0
      
      # Linear interpolation
      if len(batch_sizes) < 2:
         return self.profile_data[model_name][batch_sizes[0]]['l2_util'] / 100.0
      
      l2_utils = [self.profile_data[model_name][bs]['l2_util'] for bs in batch_sizes]
      l2_util = float(np.interp(batch_size, batch_sizes, l2_utils)) / 100.0
      
      return max(0.0, min(1.0, l2_util))
   
   def calculate_interference_factor(self, models_on_gpu):
      """
      Calculate interference factor for models sharing a GPU.
      
      Actual_Latency = Base_Latency × interference_factor
      
      Current model uses a linear regression fit:
         interference_factor = bias
            + w_l2_m1 * l2_util_1 + w_l2_m2 * l2_util_2
            + w_dram_m1 * dram_util_1 + w_dram_m2 * dram_util_2
      
      Coefficients (w_*) and bias are loaded from int_model_constant.csv.
      
      Args:
         models_on_gpu: List of (model_name, batch_size) tuples
      
      Returns:
         interference_factor: Multiplicative factor for latency
      """
      if len(models_on_gpu) <= 1:
         return 1.0  # No interference with single model
      
      per_model_stats = {}
      for model_name, bs in models_on_gpu:
         l2_util = self.get_l2_util(model_name, bs)
         _, dram_util = self.get_c_req_m_req(model_name, bs)
         stats = per_model_stats.setdefault(
            model_name, {"l2": 0.0, "dram": 0.0, "count": 0}
         )
         stats["l2"] += l2_util
         stats["dram"] += dram_util
         stats["count"] += 1
      
      if len(per_model_stats) <= 1:
         return 1.0
      
      aggregated_models = []
      for model_name, stats in per_model_stats.items():
         count = stats["count"] if stats["count"] > 0 else 1
         avg_l2 = stats["l2"] / count
         avg_dram = stats["dram"] / count
         heaviness = avg_l2 + avg_dram
         l2_used = max(0.0, min(1.0, avg_l2))
         dram_used = max(0.0, min(1.0, avg_dram))
         aggregated_models.append((model_name, l2_used, dram_used, heaviness))
      
      aggregated_models.sort(key=lambda item: item[3], reverse=True)
      model1_name, l2_util_1, dram_util_1, _ = aggregated_models[0]
      model2_name, l2_util_2, dram_util_2, _ = aggregated_models[1]
      
      # Calculate interference factor
      α1 = self.interference_constants['l2_util_coef1']
      α2 = self.interference_constants['l2_util_coef2']
      β1 = self.interference_constants['dram_util_coef1']
      β2 = self.interference_constants['dram_util_coef2']
      constant = self.interference_constants['constant']
      
      interference_factor = (constant + 
                            α1 * l2_util_1 + α2 * l2_util_2 +
                            β1 * dram_util_1 + β2 * dram_util_2)
      
      # Clamp to reasonable range
      return max(1.0, min(3.0, interference_factor))

def compute_cl_m(workload, model_name, slo_ms, estimator):
   """
   Compute minimum replication degree (cl_m) for a model while honoring the SLO.
   Evaluates available batch sizes, discarding those that violate the latency SLO
   or exceed a single GPU's memory budget.
   """
   candidate_batch_sizes = estimator.get_available_batch_sizes(model_name)
   if not candidate_batch_sizes:
      candidate_batch_sizes = [4, 8, 16, 32, 64, 128]
   
   best_bs = None
   best_throughput = 0.0
   
   for bs in candidate_batch_sizes:
      c_req, m_req = estimator.get_c_req_m_req(model_name, bs)
      if m_req >= 0.9:
         continue
      
      latency_ms = estimator.get_latency(model_name, bs)
      if latency_ms > slo_ms:
         continue
      
      throughput = (bs / latency_ms) * 1000.0
      if throughput > best_throughput:
         best_throughput = throughput
         best_bs = bs
   
   if best_bs is None:
      fallback_bs = min(candidate_batch_sizes)
      print(f"[WARN] {model_name}: no batch size satisfies SLO {slo_ms} ms "
            f"with m_req < 0.9; falling back to BS={fallback_bs}")
      latency_ms = estimator.get_latency(model_name, fallback_bs)
      best_throughput = (fallback_bs / latency_ms) * 1000.0
      best_bs = fallback_bs
   
   required_gpus = int(np.ceil(workload.rps / max(best_throughput, 1e-6)))
   return max(1, required_gpus)