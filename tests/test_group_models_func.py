import math
from mimetypes import init
import pytest
import sys
from pathlib import Path

# Add project root to Python path so we can import src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.main import group_models

class DummyWorkload:
   def __init__(self, model_name) -> None:
      self.model_name = model_name

   def __repr__(self) -> str:
      return f"DummyWorkload({self.model_name})"

class DummyEstimator:
   def __init__(self, c_m_map) -> None:
      self.c_m_map = c_m_map
   
   def compute_average_c_req_m_req(self, model_name, max_gpu_mem):
      return self.c_m_map[model_name]

def test_group_models_basic():
   workloads = [
      DummyWorkload("resnet50"),
      DummyWorkload("vgg11"),
      DummyWorkload("whisper"),
      DummyWorkload("gpt"),
      DummyWorkload("mobilenet_v3_large")
   ]

   estimator = DummyEstimator({
      "resnet50": (0.8, 0.2),
      "vgg11": (0.2, 0.8),
      "whisper": (0.7, 0.3),
      "gpt": (0.3, 0.7),
      "mobilenet_v3_large": (0.5, 0.5)
   })

   gpu_specs = [{
      "type": "test_gpu",
      "mem_mb": 24000
   }]

   max_models_per_group = 4

   # Run the function
   groups = group_models(workloads, estimator, gpu_specs, max_models_per_group)

   out = [wl.model_name for g in groups for wl in g]
   expected = [wl.model_name for wl in workloads]
   assert sorted(out) == sorted(expected)

   print([wl.model_name for wl in workloads])
   print(groups)

if __name__ == "__main__":
   test_group_models_basic()