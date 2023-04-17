import copy
from enum import Enum
from itertools import count
import threading
import logging
import time
from collections import deque

from geopy import Point

from multitask_model import MultiTaskModel


class ServerNonExistentException(Exception):
    pass


class ServerStatus(Enum):
    OFFLINE = 0
    ONLINE = 1
    STOPPING = 2


class ExpiringSet:
    def __init__(self, duration=10):
        self.duration = duration
        self.queue = deque()
        self.list = list()

    def add(self, item):
        current_time = time.time()
        self.queue.append((item, current_time))
        self.list.append(item)
        self._remove_expired_items()

    def _remove_expired_items(self):
        current_time = time.time()
        while self.queue and (current_time - self.queue[0][1]) > self.duration:
            expired_item, _ = self.queue.popleft()
            self.list.remove(expired_item)

    def __contains__(self, item):
        self._remove_expired_items()
        return item in self.list

    def __repr__(self):
        self._remove_expired_items()
        return repr(self.list)
    
    def __iter__(self):
        self._remove_expired_items()
        return iter(self.list)


# NOTE: currently we do not simulate latency in updating and querying the DHT
class DistributedHashTable:

    INFO_TYPES = ['ip', 'port', 'location', 'stages', 'queue_length', 'request_rate', 'status']

    def __init__(self, model: MultiTaskModel):
        self.lock = threading.Lock()
        self.server_info = dict()
        self.model = model
        self.stage_req_rate = ExpiringSet(duration=10)

    def update_stage_req_rate(self, server_stage_req_rate: dict):
        with self.lock:
            self.stage_req_rate.add(server_stage_req_rate)

    def get_normalized_stage_req_rate(self):
        with self.lock:
            # aggregate the request rate of each stage from all servers
            stage_req_rate = dict.fromkeys(self.model.stages, 0.0)
            for server_stage_req_rate in self.stage_req_rate:
                for stage, rate in server_stage_req_rate.items():
                    stage_req_rate[stage] += rate

            # edge case: the sum of stage_req_rate is zero, all stages have the same request rate
            total_req_rate = sum(stage_req_rate.values())
            if total_req_rate == 0:
                return {stage: 1/len(self.model.stages) for stage in self.model.stages}
            
            # normalize the request rate
            for stage, rate in stage_req_rate.items():
                stage_req_rate[stage] = rate / total_req_rate
            
            return stage_req_rate


    # get specific information of a server
    def get_server_info(self, server_id: int, info_type: str):
        with self.lock:
            if server_id not in self.server_info:
                raise ServerNonExistentException
            
            if info_type not in self.INFO_TYPES:
                raise Exception('Invalid info type')

            return self.server_info[server_id][info_type]

    # get all information of a server
    def get_server_info_all(self, server_id: int):
        with self.lock:
            return copy.deepcopy(self.server_info[server_id])

    # modify specific information of a server
    def modify_server_info(self, server_id: int, info_type: str, value):
        # logging.debug(f"DHT updating: Server {server_id} {info_type} = {value}.")
        with self.lock:
            if server_id not in self.server_info:
                raise ServerNonExistentException
        
            if info_type not in self.INFO_TYPES:
                raise Exception('Invalid info type')

            # Update the specific information of an existing server
            self.server_info[server_id][info_type] = value

    # initialize a new server entry in the DHT
    # the status will be turned to ONLINE when the server first updates its info
    def register_server(self) -> int:
        with self.lock:

            # Find the first available server_id
            logging.debug(f"Getting new server id...")
            server_id = 0
            for server_id in count():
                if server_id not in self.server_info:
                    break
            logging.debug(f"Found a new server id {server_id}.")

            self.server_info[server_id] = {
                'ip': None,
                'port': None,
                'location': None,
                'stages': [],
                'queue_length': 0,
                'request_rate': 0,
                'status': ServerStatus.OFFLINE
            }

            return server_id

    # delete a server entry from the DHT
    def delete_server(self, server_id: int):
        with self.lock:            
            if server_id not in self.server_info:
                raise Exception('Server does not exist')

            # Delete the entire server entry
            del self.server_info[server_id]

    """
    Servers often need to contact DHT in order to get a list of servers serving
    the stage they need right now, for a task they are executing. This method
    allows for that by returning the set of such servers.
    """
    def get_servers_with_stage(self, stage_name: str) -> list[int]:
        # TODO: Is using the lock necessary here? If we mess up, and return a
        # server that does not in fact serve the given stage, we should be fine
        # since the server will have a timeout, and just try again.
        output = list()
        with self.lock:
            for server, info in self.server_info.items():
                if info["status"] != ServerStatus.ONLINE:
                    continue
                stages_served = info["stages"]
                if stages_served != None and stage_name in stages_served:
                    output.append(server)
        return output

    def get_server_location(self, server_id: int) -> Point:
        return self.get_server_info(server_id, 'location')

    def get_number_of_servers(self):
        with self.lock:
            return len(self.server_info)
    
    def get_server_request_rate(self, server_id: int):
        return self.get_server_info(server_id, 'request_rate')
    
    def get_server_queue_length(self, server_id: int):
        return self.get_server_info(server_id, 'queue_length')

    def get_server_ip_port(self, server_id: int):
        with self.lock:
            if server_id not in self.server_info:
                raise ServerNonExistentException
            return self.server_info[server_id]['ip'], self.server_info[server_id]['port']
    
    def get_server_id_by_ip_port(self, ip, port):
        with self.lock:
            for server_id, server_info in self.server_info.items():
                if server_info['ip'] == ip and server_info['port'] == port:
                    return server_id
        return None
    
    def get_server_status(self, server_id: int):
        return self.get_server_info(server_id, 'status')

    def to_json(self):
        with self.lock:
            return copy.deepcopy(self.server_info)
