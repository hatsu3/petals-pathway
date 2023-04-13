import copy
from enum import Enum
from itertools import count
import threading
import logging

from geopy import Point


class ServerNonExistentException(Exception):
    pass

class ServerStatus(Enum):
    OFFLINE = 0
    ONLINE = 1


# NOTE: currently we do not simulate latency in updating and querying the DHT
class DistributedHashTable:

    INFO_TYPES = ['ip', 'port', 'location', 'stages', 'load', 'status']

    def __init__(self):
        self.lock = threading.Lock()
        self.server_info = dict()

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
        # logging.info(f"DHT updating: Server {server_id} {info_type} = {value}.")
        with self.lock:
            if server_id not in self.server_info:
                raise Exception('Server does not exist')
        
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
                'status': None,
                'stages': None,
                'load': None,
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
                stages_served = info["stages"]
                if stages_served != None and stage_name in stages_served:
                    output.append(server)
        return output

    def get_server_location(self, server_id: int) -> Point:
        return self.get_server_info(server_id, 'location')

    def get_number_of_servers(self):
        with self.lock:
            return len(self.server_info)
    
    def get_server_load(self, server_id: int):
        return self.get_server_info(server_id, 'load')

    def get_server_ip_port(self, server_id: int):
        return self.get_server_info(server_id, 'ip'), self.get_server_info(server_id, 'port')
    
    def get_server_id_by_ip_port(self, ip, port):
        with self.lock:
            for server_id, server_info in self.server_info.items():
                if server_info['ip'] == ip and server_info['port'] == port:
                    return server_id
        return None

    def to_json(self):
        with self.lock:
            return copy.deepcopy(self.server_info)
