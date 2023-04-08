import json
from pathlib import Path

import torch
import torch.nn as nn


# this class is a symbolic representation of the real module
# we expect the caller to encapsulate all modules in a stage into a single module
class Stage:
    def __init__(self, name: str, module_cls, *args, **kwargs):
        self.name = name
        self.module_cls = module_cls
        self.args = args
        self.kwargs = kwargs
        self.module = None

        if not issubclass(module_cls, nn.Module):
            raise ValueError("Module class must be a subclass of nn.Module")

    @property
    def instantiated(self):
        return self.module is not None
    
    @property
    def module_type(self):
        return self.module_cls.__name__
    
    # build the real module in CPU memory in training mode
    # the caller is responsible for moving the module to GPU and setting the correct mode
    def build(self):
        if self.instantiated:
            raise ValueError("Module has already been instantiated")
        self.module = self.module_cls(*self.args, **self.kwargs)

    # the type and location of the output depends on the module
    # the caller is responsible for data moving and type conversion
    def forward(self, *data: torch.Tensor):
        if not self.instantiated:
            raise ValueError("Module has not been instantiated")
        assert isinstance(self.module, nn.Module)
        output = self.module(*data)
        return output
    
    # save the stage information and optionally the module weights
    def save(self, save_file: str, save_weights: bool = False):
        save_path = Path(save_file)

        if save_weights and self.instantiated:
            assert isinstance(self.module, nn.Module)
            weights = self.module.state_dict()
        else:
            weights = None

        state_dict = {
            "stage_info": {
                "name": self.name,
                "module_type": self.module_type,
                "args": self.args,
                "kwargs": self.kwargs,
            },
            "weights": weights,
        }

        torch.save(state_dict, save_path)

    @classmethod
    def load(cls, save_file: str, cls_map: dict[str, type]):
        save_path = Path(save_file)
        state_dict = torch.load(save_path)

        stage_info = state_dict["stage_info"]
        weights = state_dict["weights"]

        if stage_info["module_type"] not in cls_map:
            raise ValueError(f"Module type {stage_info['module_type']} is not in the class map")

        stage = cls(
            name=stage_info["name"],
            module_cls=cls_map[stage_info["module_type"]],
            *stage_info["args"],
            **stage_info["kwargs"],
        )

        if weights is not None:
            stage.build()
            assert isinstance(stage.module, nn.Module)
            stage.module.load_state_dict(weights)
        
        return stage


# a path is a list of stages executed in sequence
# this class does not contain any SLA information or request rate
class TaskPath:
    def __init__(self, name: str, stages: list[Stage]):
        self.name = name
        self.stages = stages

    @property
    def instantiated(self):
        return all(stage.instantiated for stage in self.stages)
    
    def build(self):
        for stage in self.stages:
            if not stage.instantiated:
                stage.build()

    def forward(self, *data: torch.Tensor):
        for stage in self.stages:
            data = stage.forward(*data)
        return data


# NOTE: currently we assume the model is not updated, otherwise we will need to publish it to DHT
# a multi-task model is basically a collection of paths with shared stages
# each stage and task is assigned a numerical ID, but we can also use readable names later
class MultiTaskModel:
    def __init__(self, stages: list[Stage], paths: list[TaskPath]):
        self.stages = {stage.name: stage for stage in stages}
        self.paths = {task.name: task for task in paths}

        # check names are unique
        if len(self.stages) != len(stages):
            raise ValueError("Stage names must be unique")
        if len(self.paths) != len(paths):
            raise ValueError("Task names must be unique")

        if set(self.stages) != set([
            stage for path in self.paths.values()
            for stage in path.stages
        ]):
            raise ValueError("Some stages are not used or some tasks use stages that are not provided")

        self.routing_table = self._build_routing_table()

    # routing table is somehow like an adjacency list representation of the multi-task model
    # maps stage name to a dict that maps path/task name to the next stage name
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

    # get the next stage name given the current stage name and task name
    def get_next_stage(self, stage_name: str, task_name: str):
        if stage_name not in self.routing_table:
            raise ValueError(f"Stage {stage_name} does not exist")
        
        if task_name not in self.routing_table[stage_name]:
            # case 1. current stage is the last stage of the task
            if self.stages[stage_name] in self.paths[task_name].stages:
                return None
            else:
                raise ValueError(f"Task {task_name} does not use stage {stage_name}")

        # case 2. current stage is not the last stage of the task
        return self.routing_table[stage_name][task_name]

    # WARNING: this function should not be called
    def add_path(self, path: TaskPath):
        if path.name in self.paths:
            raise ValueError(f"Task name {path.name} already exists")
        
        for stage in path.stages:
            if stage.name in self.stages and stage is not self.stages[stage.name]:
                raise ValueError(f"Stage name {stage.name} already exists and the stage is different")
        
        self.paths[path.name] = path

        for stage in path.stages:
            if stage.name not in self.stages:
                self.stages[stage.name] = stage
        
        self.routing_table = self._build_routing_table()

    # NOTE: this function is not used in simulation
    # the caller is responsible for moving the modules and data to GPU and setting the correct mode
    def forward(self, task_name: str, *data: torch.Tensor):
        if task_name not in self.paths:
            raise ValueError(f"Task {task_name} does not exist")

        path = self.paths[task_name]
        result = path.forward(*data)
        return result

    def save(self, save_dir: str, save_weights: bool = False):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=False)

        # save all stages
        ckpt_path = save_path / "stages"
        ckpt_path.mkdir(parents=True, exist_ok=False)

        for stage in self.stages.values():
            save_stage_weights = save_weights and stage.instantiated
            stage_save_path = ckpt_path / (stage.name + ".pt")
            stage.save(str(stage_save_path), save_weights=save_stage_weights)
        
        # save topology as a json file
        topology = {
            path.name: [stage.name for stage in path.stages]
            for path in self.paths.values()
        }

        with open(save_path / "topology.json", "w") as f:
            json.dump(topology, f)

    @classmethod
    def load(cls, load_dir: str, cls_map: dict[str, type]):
        load_path = Path(load_dir)

        # load all stages
        ckpt_path = load_path / "stages"
        stage_dict = dict()
        for ckpt in ckpt_path.glob("*.pt"):
            stage = Stage.load(str(ckpt), cls_map)
            stage_dict[stage.name] = stage

        # load topology
        with open(load_path / "topology.json", "r") as f:
            topology = json.load(f)

        paths = list()
        for path_name, stage_names in topology.items():
            stages = [stage_dict[stage_name] for stage_name in stage_names]
            paths.append(TaskPath(path_name, stages))

        return cls(list(stage_dict.values()), paths)
