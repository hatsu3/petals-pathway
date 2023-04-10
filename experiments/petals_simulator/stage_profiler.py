import json

import torch

from .multitask_model import MultiTaskModel, Stage


def _profile_module(module, input_shape, batch_size, num_warmup=10, num_repeats=100, num_iters=10):
    # Set up input tensor with given batch size
    input_tensor = torch.randn(batch_size, *input_shape).cuda()

    # Set module to inference mode and use torch.no_grad()
    module.eval()

    with torch.no_grad():
        # Warmup the GPU and PyTorch by executing the module a few times
        for i in range(num_warmup):
            _ = module(input_tensor)

        # Measure the latency using CUDA events
        latencies = []
        for i in range(num_repeats):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record() # type: ignore
            for j in range(num_iters):
                _ = module(input_tensor)
            end_event.record() # type: ignore

            torch.cuda.synchronize()
            latency_ms = start_event.elapsed_time(end_event) / num_iters
            latencies.append(latency_ms)

    # Sort the latencies and discard the top and bottom 10% to avoid outliers
    latencies.sort()
    num_discard = int(len(latencies) * 0.1)
    latencies = latencies[num_discard:-num_discard]

    # Return the average latency in milliseconds
    avg_latency_ms = sum(latencies) / len(latencies)
    return avg_latency_ms


class StageProfiler:
    def __init__(self, batch_sizes, num_warmup=10, num_repeats=100, num_iters=10):
        self.batch_sizes = batch_sizes
        self.prof_kwargs = {
            'num_warmup': num_warmup,
            'num_repeats': num_repeats,
            'num_iters': num_iters
        }
        self._results = list()

    def profile_stage(self, stage: Stage, input_shape: tuple[int]):
        print(f"Profiling {stage.name}...")
        
        if not stage.instantiated:
            stage.build()
            assert isinstance(stage.module, torch.nn.Module)
            stage.module.cuda()

        for batch_size in self.batch_sizes:
            print(f"- Batch size: {batch_size}")
            latency_ms = _profile_module(stage.module, input_shape, batch_size, **self.prof_kwargs)
            self._results.append({
                'name': stage.name,
                'type': stage.module.__class__.__name__,
                'batch_size': batch_size,
                'latency_ms': latency_ms
            })
    
    # TODO: support symbolic execution to get input shape
    def profile_model(self, model: MultiTaskModel):
        """for stage in model.stages.values():
            self.profile_stage(stage, stage.input_shape)"""

    def get_profile_results(self):
        return ProfilingResults(self._results, self.batch_sizes, self.prof_kwargs)


# Profiling results for all stages in a multi-stage model
# and profiling configurations
class ProfilingResults:
    def __init__(self, results: list[dict], batch_sizes: list[int], prof_kwargs: dict):
        self.results = results
        self.batch_sizes = batch_sizes
        self.prof_kwargs = prof_kwargs

    def save(self, json_file: str):
        json.dump({
            'results': self.results,
            'batch_sizes': self.batch_sizes,
            'prof_kwargs': self.prof_kwargs
        }, open(json_file, 'w'))

    @classmethod
    def load(cls, json_file: str):
        data = json.load(open(json_file, 'r'))
        return cls(data['results'], data['batch_sizes'], data['prof_kwargs'])

    # Get the execution latency of a stage in milliseconds
    def get_latency(self, stage_name: str, batch_size: int):
        for result in self.results:
            if result['name'] == stage_name and result['batch_size'] == batch_size:
                return result['latency_ms']
        raise ValueError(f"Cound not find results for stage {stage_name} with batch size {batch_size}")
