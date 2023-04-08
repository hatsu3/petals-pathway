import os

import pandas as pd
import torch


def profile_module(module, input_shape, batch_size, num_warmup=10, num_repeats=100, num_iters=10):
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

    def profile(self, module_name, module, input_shape):
        print(f"Profiling {module_name}...")
        for batch_size in self.batch_sizes:
            print(f"- Batch size: {batch_size}")
            latency_ms = profile_module(module, input_shape, batch_size, **self.prof_kwargs)
            self._results.append({
                'name': module_name,
                'type': module.__class__.__name__,
                'batch_size': batch_size,
                'latency_ms': latency_ms
            })

    def to_csv(self, csv_file, append=False):
        if append and os.path.exists(csv_file):
            results_df = pd.read_csv(csv_file)
            results_df = results_df.append(self._results, ignore_index=True)
        else:
            results_df = pd.DataFrame(self._results)

        results_df.to_csv(csv_file, index=False)
        print(f"Saved profiling results to {csv_file}")


# Profiling results for all stages in a multi-stage model
# and profiling configurations
class ProfilingResults:
    def __init__(self) -> None:
        pass

    def save(self, json_file: str):
        pass
    
    @classmethod
    def load(cls, json_file: str):
        pass

    # Get the execution latency of a stage in milliseconds
    def get_latency(self, stage_name: str, batch_size: int):
        return 0.0


if __name__ == '__main__':
    import torch.nn as nn
    from torchvision.models import resnet18

    # Create a ResNet-18 model
    model = resnet18(pretrained=False).cuda()

    # Create a dummy input tensor
    input_tensor = torch.randn(1, 3, 224, 224).cuda()

    # Create a dummy module
    dummy_module = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(512, 1000)
    ).cuda()

    # Create a profiler
    profiler = StageProfiler(batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128])

    # Profile the ResNet-18 model
    profiler.profile('ResNet-18', model, input_shape=(3, 224, 224))

    # Profile the dummy module
    profiler.profile('Dummy Module', dummy_module, input_shape=(3, 224, 224))

    # Save the profiling results to a CSV file
    profiler.to_csv('data/profiling_results.csv')
