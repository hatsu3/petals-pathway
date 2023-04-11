import copy
from itertools import count
import threading

from geopy import Point


# NOTE: currently we do not simulate latency in updating and querying the DHT
class DistributedHashTable:

    INFO_TYPES = ['ip', 'port', 'location', 'stages', 'load']

    def __init__(self):
        self.lock = threading.Lock()
        self.server_info = dict()

    # get specific information of a server
    def get_server_info(self, server_id: int, info_type: str):
        with self.lock:
            if server_id not in self.server_info:
                raise Exception('Server does not exist')
            
            if info_type not in self.INFO_TYPES:
                raise Exception('Invalid info type')

            return self.server_info[server_id][info_type]

    # get all information of a server
    def get_server_info_all(self, server_id: int):
        with self.lock:
            return copy.deepcopy(self.server_info[server_id])

    # modify specific information of a server
    def modify_server_info(self, server_id: int, info_type: str, value):
        with self.lock:
            if server_id not in self.server_info:
                raise Exception('Server does not exist')
        
            if info_type not in self.INFO_TYPES:
                raise Exception('Invalid info type')

            # Update the specific information of an existing server
            self.server_info[server_id][info_type] = value

    # initialize a new server entry in the DHT
    def add_server(self, server_id: int):
        with self.lock:
            if server_id in self.server_info:
                raise Exception('Server already exists')
            else:
                self.server_info[server_id] = {
                    'ip': None,
                    'port': None,
                    'location': None,
                    'status': None,
                    'stages': None,
                    'load': None
                }

    # delete a server entry from the DHT
    def delete_server(self, server_id: int):
        with self.lock:            
            if server_id not in self.server_info:
                raise Exception('Server does not exist')

            # Delete the entire server entry
            del self.server_info[server_id]

    # generate an id for a new server
    # return the smallest integer that is not used as a server id
    # note that some server ids may be deleted, and we will reuse them
    def get_new_server_id(self):
        with self.lock:
            for i in count():
                if i not in self.server_info:
                    return i

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
            for server in self.server_info:
                stages_served = server["stages"]
                if stage_name in stages_served:
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
