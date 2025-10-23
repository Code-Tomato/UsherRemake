from objects_dataclass import ModelProfile, Model, Classification, WorkloadRequest
import json
import os

def parse_profile(gpu_type='4090'):
   profile_path = f'data/{gpu_type}/profile.csv'
   if not os.path.exists(profile_path):
      profile_path = 'data/4090/profile.csv'  # Fallback
   with open(profile_path, 'r') as file:
      lines = file.readlines()

   model_profiles = []
   for line in lines[1:]:
      bench, batch, avg_duration, SM_util, l2_util, mem_util, achieved_occupancy, theoretical_occupancy, _ = line.split(',')
      bench = bench.strip()
      batch = int(batch.strip())
      avg_duration = float(avg_duration.strip())
      SM_util = float(SM_util.strip())
      l2_util = float(l2_util.strip())
      mem_util = float(mem_util.strip())
      achieved_occupancy = float(achieved_occupancy.strip())
      theoretical_occupancy = float(theoretical_occupancy.strip())
      model_profiles.append(ModelProfile(bench, batch, avg_duration, SM_util, l2_util, mem_util, achieved_occupancy, theoretical_occupancy))
   return model_profiles

def list_models(model_profiles: list[ModelProfile]):
   models = []
   for model_profile in model_profiles:
      c_req, m_req = model_profile.SM_util/100, model_profile.mem_util/100
      ratio = c_req/m_req if m_req != 0 else 0
      if ratio > 1.2:
         classify = Classification.C_HEAVY
      elif ratio < .8:
         classify = Classification.M_HEAVY
      else:
         classify = Classification.BALANCED
      models.append(Model(model_profile.name, model_profile.bs, c_req, m_req, classify, model_profile.duration/1000, model_profile.bs/(model_profile.duration/1000)))
   return models

def load_gpu_config():
   """Load GPU configurations from device-config.json"""
   with open('config/device-config.json', 'r') as f:
      config = json.load(f)
   return config['device_specs']

def load_model_memory_config():
   """Load model memory requirements from mem-config.json"""
   with open('config/mem-config.json', 'r') as f:
      config = json.load(f)
   model_mem = {}
   for model in config['models']:
      model_mem[model['name']] = model['mem']
   return model_mem

def parse_input_workload(input_file='input.csv'):
   """Parse input.csv workload specification"""
   model_mem = load_model_memory_config()
   model_names = list(model_mem.keys())
   
   with open(input_file, 'r') as f:
      lines = f.readlines()
   
   num_models = int(lines[0].strip())
   workloads = []
   
   for i in range(1, num_models + 1):
      parts = [p.strip() for p in lines[i].split(',') if p.strip()]
      model_id = int(parts[0])
      rps = float(parts[1])
      slo_ms = float(parts[2])
      
      # Map model_id to model_name (assuming 1-indexed in input)
      if 0 <= model_id - 1 < len(model_names):
         model_name = model_names[model_id - 1]
      else:
         model_name = f"model_{model_id}"
      
      workloads.append(WorkloadRequest(model_id, model_name, rps, slo_ms))
   
   return workloads

if __name__ == "__main__":
   model_profiles = parse_profile()
   models = list_models(model_profiles)
   for model in models:
      print(model)