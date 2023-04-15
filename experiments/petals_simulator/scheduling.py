from abc import ABC, abstractmethod
import random
import time

from multitask_model import MultiTaskModel
from utils import GPUTask
from stage_profiler import ProfilingResults


class SchedulingPolicy(ABC):
    
    """A scheduling policy that determines the priority of a task"""

    def __init__(self, model: MultiTaskModel):
        self.model = model

    @abstractmethod
    def calculate_priority(self, task: GPUTask) -> float:
        pass


class RandomSchedulingPolicy(SchedulingPolicy):

    """A baseline scheduling policy that randomly assigns priorities"""

    def calculate_priority(self, task: GPUTask) -> float:
        return random.uniform(0, 100)


class LatencyAwareSchedulingPolicy(SchedulingPolicy):
    def __init__(self, model: MultiTaskModel, profiling_results: ProfilingResults):
        super().__init__(model)
        self.profiling_results = profiling_results
    
    def _estimate_time_to_completion(self, task: GPUTask):
        estimation = 0.0
        task_name = task.request.task_name
        stage_name = self.model.get_stage(task_name, task.request.next_stage_idx).name
        while stage_name is not None:
            estimation += self.profiling_results.get_latency(stage_name, batch_size=1)
            stage_name = self.model.get_next_stage(stage_name, task_name)
            
        return estimation
    
    def calculate_priority(self, task: GPUTask) -> float:
        # estimated_completion_time = current_time - timestamp + estimate_time_to_completion
        # the lower priority, the earlier to be scheduled, so negate this expression
        return -(time.time() / 1e3 - task.request.timestamp / 1e3 + self._estimate_time_to_completion(task))
