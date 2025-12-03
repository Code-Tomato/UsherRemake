from objects_dataclass import WorkloadRequest
import json

def load_gpu_config():
   """Load GPU configurations from device-config.json"""
   with open('config/device-config.json', 'r') as f:
      config = json.load(f)
   return config['device_specs']

def get_available_model_names(gpu_type='4090'):
   """Get list of available model names from profile.csv"""
   profile_path = f'data/{gpu_type}/profile.csv'
   
   model_names_set = set()
   try:
      with open(profile_path, 'r') as f:
         lines = f.readlines()
      
      # Skip header, extract model names from first column
      for line in lines[1:]:
         if line.strip():
            model_name = line.split(',')[0].strip()
            if model_name:
               model_names_set.add(model_name)
   except Exception as e:
      print(f"Warning: Could not load model names from {profile_path}: {e}")
   
   # Return sorted list for consistent ordering
   return sorted(list(model_names_set))

def parse_input_workload(input_file='input.csv', gpu_type='4090'):
   """Parse input.csv workload specification
   
   Format: CSV with header row (Model,RPS,SLO)
   """
   model_names = get_available_model_names(gpu_type)
   
   with open(input_file, 'r') as f:
      lines = [line.strip() for line in f.readlines() if line.strip()]
   
   if not lines:
      raise ValueError(f"Input file {input_file} is empty")
   
   # Skip header row
   data_lines = lines[1:]
   
   workloads = []
   for line in data_lines:
      parts = [p.strip() for p in line.split(',') if p.strip()]
      if len(parts) < 3:
         continue  # Skip invalid lines
      
      model_name = parts[0]
      rps = float(parts[1])
      slo_ms = float(parts[2])
      
      if model_name not in model_names:
         raise ValueError(f"Unknown model name: '{model_name}'. "
                          f"Expected a model name from {model_names}")
      
      # Find the ID (1-indexed)
      model_id = model_names.index(model_name) + 1
      workloads.append(WorkloadRequest(model_id, model_name, rps, slo_ms))
   
   return workloads