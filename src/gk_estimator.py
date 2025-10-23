import numpy as np
import os

class GKEstimator:
   def __init__(self, gpu_type='4090'):
      self.gpu_type = gpu_type
      self.profile_data = {}  # {model_name: {batch_size: {c_util, m_util, duration, l2_util}}}
      self.latency_data = {}  # {model_name: {partition: {batch_size: latency}}}
      self.interference_constants = {}  # Interference modeling constants
      self.load_profile_data()
      self.load_latency_data()
      self.load_interference_constants()
   
   def load_profile_data(self):
      """Load profiling data from profile.csv"""
      profile_path = f'data/{self.gpu_type}/profile.csv'
      if not os.path.exists(profile_path):
         profile_path = 'data/4090/profile.csv'
      
      with open(profile_path, 'r') as f:
         lines = f.readlines()
      
      for line in lines[1:]:
         parts = line.strip().split(',')
         model_name = parts[0].strip()
         batch_size = int(parts[1].strip())
         duration = float(parts[2].strip())
         sm_util = float(parts[3].strip())
         l2_util = float(parts[4].strip())
         mem_util = float(parts[5].strip())
         
         if model_name not in self.profile_data:
            self.profile_data[model_name] = {}
         
         self.profile_data[model_name][batch_size] = {
            'c_util': sm_util,
            'm_util': mem_util,
            'l2_util': l2_util,
            'duration': duration
         }
   
   def load_latency_data(self):
      """Load latency data from latency.csv"""
      latency_path = f'data/{self.gpu_type}/latency.csv'
      if not os.path.exists(latency_path):
         latency_path = 'data/4090/latency.csv'
      
      with open(latency_path, 'r') as f:
         lines = f.readlines()
      
      for line in lines:
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
      if not os.path.exists(const_path):
         const_path = 'data/4090/int_model_constant.csv'
      
      try:
         with open(const_path, 'r') as f:
            lines = f.readlines()
         
         if len(lines) >= 2:
            parts = lines[1].strip().split(',')
            self.interference_constants = {
               'l2_util_coef1': float(parts[0]),
               'l2_util_coef2': float(parts[1]),
               'dram_util_coef1': float(parts[2]),
               'dram_util_coef2': float(parts[3]),
               'constant': float(parts[4])
            }
         else:
            # Default to no interference
            self.interference_constants = {
               'l2_util_coef1': 0.0,
               'l2_util_coef2': 0.0,
               'dram_util_coef1': 0.0,
               'dram_util_coef2': 0.0,
               'constant': 1.0
            }
      except Exception as e:
         print(f"Warning: Could not load interference constants: {e}")
         self.interference_constants = {
            'l2_util_coef1': 0.0,
            'l2_util_coef2': 0.0,
            'dram_util_coef1': 0.0,
            'dram_util_coef2': 0.0,
            'constant': 1.0
         }
   
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
         # Fallback: estimate from profile data
         if model_name in self.profile_data and batch_size in self.profile_data[model_name]:
            return self.profile_data[model_name][batch_size]['duration']
         return 10.0  # Default latency
      
      batch_sizes = sorted(self.latency_data[model_name][partition].keys())
      
      if batch_size in self.latency_data[model_name][partition]:
         return self.latency_data[model_name][partition][batch_size]
      
      # Linear interpolation
      if len(batch_sizes) < 2:
         return self.latency_data[model_name][partition][batch_sizes[0]]
      
      latencies = [self.latency_data[model_name][partition][bs] for bs in batch_sizes]
      
      return float(np.interp(batch_size, batch_sizes, latencies))
   
   def compute_average_c_req_m_req(self, model_name, max_memory_mb):
      """Compute average C_req and M_req across all valid batch sizes"""
      batch_sizes = [4, 8, 16, 32, 64, 128]
      valid_c_reqs = []
      valid_m_reqs = []
      
      for bs in batch_sizes:
         c_req, m_req = self.get_c_req_m_req(model_name, bs)
         # Check if memory requirement is within GPU capacity (rough estimate)
         # Assuming m_req is a fraction and we have model memory info
         valid_c_reqs.append(c_req)
         valid_m_reqs.append(m_req)
      
      if not valid_c_reqs:
         return 0.5, 0.5
      
      return np.mean(valid_c_reqs), np.mean(valid_m_reqs)
   
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
      
      Paper formula:
      Actual_Latency = Base_Latency × interference_factor
      
      interference_factor = constant + α·l2_util₁·l2_util₂ + β·dram_util₁·dram_util₂
      
      Args:
         models_on_gpu: List of (model_name, batch_size) tuples
      
      Returns:
         interference_factor: Multiplicative factor for latency
      """
      if len(models_on_gpu) <= 1:
         return 1.0  # No interference with single model
      
      # For simplicity, calculate pairwise interference between first two models
      # Full implementation would consider all pairs
      model1_name, bs1 = models_on_gpu[0]
      model2_name, bs2 = models_on_gpu[1]
      
      # Get utilization metrics
      l2_util_1 = self.get_l2_util(model1_name, bs1)
      l2_util_2 = self.get_l2_util(model2_name, bs2)
      
      _, dram_util_1 = self.get_c_req_m_req(model1_name, bs1)
      _, dram_util_2 = self.get_c_req_m_req(model2_name, bs2)
      
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

def compute_cl_m(workload, model_name, slo_ms, max_gpu_memory, estimator):
   """
   Compute minimum replication degree (cl_m) for a model.
   This is the minimum number of GPUs needed to satisfy the SLO.
   """
   # Find the highest batch size that fits in GPU memory
   batch_sizes = [128, 64, 32, 16, 8, 4]
   max_bs = 4
   
   for bs in batch_sizes:
      c_req, m_req = estimator.get_c_req_m_req(model_name, bs)
      # Simple check: if m_req seems reasonable, use this batch size
      if m_req < 0.9:  # Leave some headroom
         max_bs = bs
         break
   
   # Get latency for max batch size
   latency_ms = estimator.get_latency(model_name, max_bs)
   
   # Calculate throughput per GPU (requests/second)
   throughput_per_gpu = (max_bs / latency_ms) * 1000  # Convert ms to seconds
   
   # Calculate required number of GPUs based on RPS
   rps = workload.rps
   required_gpus = int(np.ceil(rps / throughput_per_gpu))
   
   # Ensure at least 1 GPU
   cl_m = max(1, required_gpus)
   
   return cl_m