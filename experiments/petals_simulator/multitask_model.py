import json
from pathlib import Path

import torch
import torch.nn as nn


# this class is a symbolic representation of the real module
class Stage:
    def __init__(self, name: str, module: nn.Module):
        self.name = name
        self.module = module

    def process(self, data):
        output = self.module(data)
        return output
    

class TaskPath:
    def __init__(self, name: str, stages: list[Stage], request_rate: float, latency_slo: float):
        self.name = name
        self.stages = stages
        self.request_rate = request_rate
        self.latency_slo = latency_slo

    def process(self, data):
        for stage in self.stages:
            data = stage.process(data)
        return data


class MultiTaskModel:
    def __init__(self, stages: list[Stage], paths: list[TaskPath]):
        self.stages = {stage.name: stage for stage in stages}
        self.paths = {path.name: path for path in paths}

        # sanity check
        assert set(self.stages) == set([
            stage for path in self.paths.values() 
            for stage in path.stages
        ])

        # stage -> { path -> next stage }
        self.routing_table = self._build_routing_table()

    def _build_routing_table(self):
        routing_table = {}

        for path in self.paths.values():
            for i in range(len(path.stages) - 1):
                stage = path.stages[i]
                next_stage = path.stages[i + 1]

                if stage.name not in routing_table:
                    routing_table[stage.name] = {}
                routing_table[stage.name][path.name] = next_stage.name

        return routing_table

    def get_next_stage(self, stage_name: str, path_name: str):
        if stage_name not in self.routing_table:
            return None
        if path_name not in self.routing_table[stage_name]:
            return None
        return self.routing_table[stage_name][path_name]

    def add_path(self, path: TaskPath):
        """Add a new path to the model. A path is a list of stages."""
        if path.name in self.paths:
            raise ValueError("Path name already exists")
        
        self.paths[path.name] = path

        for stage in path.stages:
            if stage.name not in self.stages:
                self.stages[stage.name] = stage

    def process(self, data, path_name: str):
        if path_name not in self.paths:
            raise ValueError("Path name does not exist")

        path = self.paths[path_name]
        result = path.process(data)
        return result

    def save(self, save_dir: str):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=False)

        # save all stages
        ckpt_path = save_path / "checkpoints"
        ckpt_path.mkdir(parents=True, exist_ok=False)

        for stage in self.stages.values():
            torch.save(stage.module, ckpt_path / (stage.name + ".pt"))
        
        # save topology as a json file
        topology = {
            path.name: [stage.name for stage in path.stages]
            for path in self.paths.values()
        }

        with open(save_path / "topology.json", "w") as f:
            json.dump(topology, f)

        # save request rates and latency SLOs as a single json file
        task_info = {
            path.name: {
                "request_rate": path.request_rate,
                "latency_slo": path.latency_slo
            }
            for path in self.paths.values()
        }

        with open(save_path / "task_info.json", "w") as f:
            json.dump(task_info, f)

    @classmethod
    def load(cls, load_dir: str):
        load_path = Path(load_dir)

        # load all stages
        ckpt_path = load_path / "checkpoints"
        stages = []
        for ckpt in ckpt_path.glob("*.pt"):
            stage_name = ckpt.stem
            module = torch.load(ckpt)
            stages.append(Stage(stage_name, module))

        # load topology
        with open(load_path / "topology.json", "r") as f:
            topology = json.load(f)

        paths = []
        for path_name, stage_names in topology.items():
            stages = [stages[stage_name] for stage_name in stage_names]
            paths.append(TaskPath(path_name, stages, 0, 0))

        # load request rates and latency SLOs
        with open(load_path / "task_info.json", "r") as f:
            task_info = json.load(f)

        for path in paths:
            path.request_rate = task_info[path.name]["request_rate"]
            path.latency_slo = task_info[path.name]["latency_slo"]

        return cls(stages, paths)
