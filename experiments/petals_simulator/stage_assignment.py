
from abc import ABC, abstractmethod
import sys

from dht import DistributedHashTable
from multitask_model import MultiTaskModel, Stage


class StageAssignmentPolicy(ABC):

    """A policy that assigns stages to servers"""

    def __init__(self, model: MultiTaskModel, dht: DistributedHashTable):
        self.model = model
        self.dht = dht

    @abstractmethod
    def assign_stages(self, current_stages: list[str]) -> list[str]:
        pass


class AllToAllStageAssignmentPolicy(StageAssignmentPolicy):

    """A policy that assigns all stages to all servers"""

    def assign_stages(self, current_stages: list[str]) -> list[str]:
        return [stage.name for stage in self.model.get_stages()]
 

class UniformStageAssignmentPolicy(StageAssignmentPolicy):

    """A policy that assigns stages to servers based on 
    the number of servers that are serving the stage. 
    Ideally, each stage should have the same number of replicas"""

    def assign_stages(self, current_stages: list[str]) -> list[str]:
        # Get the number of servers that are serving each stage
        stages = self.model.get_stages()
        stage_num_replicas = {
            stage.name: len(self.dht.get_servers_with_stage(stage.name)) 
            for stage in stages
        }

        # Get the total number of servers in the swarm
        number_of_servers = self.dht.get_number_of_servers()
        assert number_of_servers > 0, "There should be at least one server in the swarm"
        
        # Serve all stages that are not being served by any server
        for stage_name, num_replicas in stage_num_replicas.items():
            if num_replicas < 1:
                assert stage_name not in current_stages
                current_stages.append(stage_name)
        
        # Calculate the average number of stages served by each server
        average_load = sum(stage_num_replicas.values()) / self.dht.get_number_of_servers()

        # If this server is serving less stages than the average, add more stages
        if average_load > len(current_stages):
            candidate_num_replicas = {
                stage: stage_num_replicas[stage]
                for stage in stage_num_replicas
                if stage not in current_stages
            }
            while average_load > len(current_stages) and len(candidate_num_replicas) != 0:
                # Pick the stage with the least number of replicas
                candidate = min(candidate_num_replicas, key=candidate_num_replicas.get) # type: ignore
                current_stages.append(candidate)
                del candidate_num_replicas[candidate]
        
        # If this server is serving more stages than the average, remove some stages
        else:
            candidate_num_replicas = {
                stage: stage_num_replicas[stage] 
                for stage in current_stages
            }
            while average_load < len(current_stages) and len(candidate_num_replicas) != 0:
                # Pick the stage with the most number of replicas
                candidate = max(candidate_num_replicas, key=candidate_num_replicas.get) # type: ignore
                # Only remove the stage if there are at least one other server serving it
                if candidate_num_replicas[candidate] > 1:
                    current_stages.remove(candidate)
                del candidate_num_replicas[candidate]
        
        return current_stages


class RequestRateStageAssignmentPolicy(StageAssignmentPolicy):

    """A policy that assigns stages to servers based on the request rate of the stage.
    Ideally, workload should be balanced across servers"""
    
    # this function is only called when a server is online or starting up (current_stages is empty)
    # which means current_stages are reflected in the DHT
    def assign_stages(self, current_stages: list[str]) -> list[str]:
        stages = self.model.get_stages()
        req_rate = self.dht.get_normalized_stage_req_rate()  # stage name -> normalized req rate

        # calculate the fulfillment score for each stage as num_replicas / req_rate
        # the lower the score, the higher priority we should allocate more servers to the stage
        def calc_fulfillment_score(stage: Stage) -> float:
            num_replicas = len(self.dht.get_servers_with_stage(stage.name))
            if req_rate[stage.name] == 0:
                return sys.maxsize  # handle zero division
            else:
                return num_replicas / req_rate[stage.name]
        
        fulfillment_scores = {
            stage.name: calc_fulfillment_score(stage) for stage in stages
        }

        number_of_servers = self.dht.get_number_of_servers()
        assert number_of_servers > 0, "There should be at least one server in the system"
        
        # we calculate the current load assuming that for each stage the load is distributed evenly
        # across all replicas
        num_stage_replicas = {
            stage.name: len(self.dht.get_servers_with_stage(stage.name)) 
            for stage in stages
        }

        current_load = sum([
            req_rate[stage] / (num_stage_replicas[stage] + 1)
            for stage in current_stages
        ])

        # serve all stages that are not currently being served by any server
        for stage_name, num_replicas in num_stage_replicas.items():
            if num_replicas == 0:
                current_stages.append(stage_name)
                current_load += req_rate[stage_name] / (num_stage_replicas[stage_name] + 1)
                del fulfillment_scores[stage_name]  # no replacement

        # our goal is to make the load of each server as close to the average load as possible
        target_load = 1 / number_of_servers
        
        # if the current load is lower than the average load, we should add more servers
        # we pick a few stages with the lowest fulfillment scores
        if target_load > current_load:
            while target_load > current_load and len(fulfillment_scores) > 0:
                candidate = min(fulfillment_scores, key=fulfillment_scores.get) # type: ignore
                current_stages.append(candidate)
                current_load += req_rate[candidate] / (num_stage_replicas[candidate] + 1)
                del fulfillment_scores[candidate]  # no replacement
        
        # if the current load is higher than the average load, we should remove some servers
        # we remove a few stages with the highest fulfillment scores
        elif target_load < current_load:
            curr_stg_scores = {
                stage: fulfillment_scores[stage] 
                for stage in current_stages
                if stage in fulfillment_scores
            }
            
            while target_load < current_load and len(curr_stg_scores) > 0:                
                candidate = max(curr_stg_scores, key=curr_stg_scores.get) # type: ignore
                # only remove the stage if there is at least one replica left
                if num_stage_replicas[candidate] >= 2:
                    current_stages.remove(candidate)
                    current_load -= req_rate[candidate] / num_stage_replicas[candidate]
                
                del curr_stg_scores[candidate]  # no replacement
        
        return current_stages
