from dataclasses import dataclass
from enum import Enum

@dataclass
class ModelProfile:
   name: str
   bs: int
   duration: float
   SM_util: float
   l2_util: float
   mem_util: float
   achieved_occupancy: float
   theoretical_occupancy: float

   def __str__(self) -> str:
      return (f"ModelProfile(name='{self.name}', bs={self.bs}, duration={self.duration:.3f}, "
               f"SM_util={self.SM_util:.2f}, l2_util={self.l2_util:.2f}, mem_util={self.mem_util:.2f}, "
               f"achieved_occupancy={self.achieved_occupancy:.2f}, theoretical_occupancy={self.theoretical_occupancy:.2f})")

class Classification(Enum):
   C_HEAVY = "c_heavy"
   M_HEAVY = "m_heavy"
   BALANCED = "balanced"

@dataclass
class Model:
   name: str
   batch_size: int
   C_req: int
   M_req: int
   classify: Classification
   t_batch: int
   goodput: float

   def __str__(self) -> str:
      return (f"Model(name='{self.name}', batch_size={self.batch_size}, C_req={self.C_req}, M_req={self.M_req}, classify={self.classify}, t_batch={self.t_batch}, goodput={self.goodput:.2f})")

@dataclass
class Group:
   models: list[Model]
   distance: float

   def compute_distance(self) -> float:
      return abs(sum(model.C_req for model in self.models) - sum(model.M_req for model in self.models))

   def __str__(self) -> str:
      return (f"Group(models=[{', '.join(str(model) for model in self.models)}], distance={self.distance:.2f})")

@dataclass
class GPU:
   gpu_id: int
   gpu_type: str
   max_compute: float  # Maximum compute capacity (SM utilization)
   max_memory: float   # Maximum memory in MB
   compute_used: float = 0.0
   memory_used: float = 0.0
   model_replicas: list = None  # List of (model_name, batch_size, replica_id)
   cost: float = 1.0  # Cost per hour
   
   def __post_init__(self):
      if self.model_replicas is None:
         self.model_replicas = []
   
   def can_fit(self, c_req: float, m_req: float) -> bool:
      return (self.compute_used + c_req <= self.max_compute and 
              self.memory_used + m_req <= self.max_memory)
   
   def remaining_space(self) -> float:
      return (self.max_compute - self.compute_used) + (self.max_memory - self.memory_used)
   
   def __str__(self) -> str:
      return (f"GPU(id={self.gpu_id}, type={self.gpu_type}, "
              f"compute={self.compute_used:.2f}/{self.max_compute:.2f}, "
              f"memory={self.memory_used:.2f}/{self.max_memory:.2f})")

@dataclass
class Assignment:
   model_name: str
   batch_size: int
   replica_id: int
   gpu_id: int
   gpu_type: str
   
   def to_dict(self, include_replica_id=False):
      result = {
         "model": self.model_name,
         "batch_size": self.batch_size,
         "gpu_id": self.gpu_id,
         "gpu_type": self.gpu_type
      }
      # Only include replica_id if it's meaningful (RD > 1 cases)
      if include_replica_id:
         result["replica_id"] = self.replica_id
      return result

@dataclass 
class WorkloadRequest:
   model_id: int
   model_name: str
   rps: float  # Requests per second
   slo_ms: float  # SLO in milliseconds
   
   def __str__(self) -> str:
      return f"WorkloadRequest(model={self.model_name}, rps={self.rps}, slo_ms={self.slo_ms})"
