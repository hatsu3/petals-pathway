import copy
import threading
from geopy import Point
from multitask_model import Stage


# TODO: rethink the design of server ID. what if two servers have the same ID?
# NOTE: currently we do not simulate latency in updating and querying the DHT
class DistributedHashTable:

    INFO_TYPES = ['ip', 'port', 'location', 'stages', 'load']

    def __init__(self):
        self.lock = threading.Lock()
        self.server_info = dict()

    def get(self, key):
        with self.lock:
            server_id, info_type = key
            
            if server_id not in self.server_info:
                raise Exception('Server does not exist')

            if info_type is None:
                return self.server_info[server_id]
            else:
                return self.server_info[server_id][info_type]

    def put(self, key, value):
        with self.lock:
            server_id, info_type = key

            if server_id not in self.server_info:
                if info_type is None:
                    # Create a new server entry with default values
                    self.server_info[server_id] = {
                        'ip': None,
                        'port': None,
                        'location': None,
                        'status': None,
                        'stages': None,
                        'load': None
                    }
                else:
                    raise Exception('Server does not exist')
            else:
                if info_type is None:
                    raise Exception('Server already exists')
                else:
                    # Update the specific information of an existing server
                    self.server_info[server_id][info_type] = value

    def delete(self, key):
        with self.lock:
            server_id, info_type = key
            
            if server_id not in self.server_info:
                raise Exception('Server does not exist')

            if info_type is None:
                # Delete the entire server entry
                del self.server_info[server_id]
            else:
                # Delete a specific piece of information of a server
                del self.server_info[server_id][info_type]

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
        return self.get((server_id, 'location'))

    def get_number_of_servers(self):
        with self.lock:
            return len(self.server_info)
    
    def get_server_load(self, server_id: int):
        return self.get((server_id, 'load'))

    def get_server_ip_port(self, server_id: int):
        return self.get((server_id, 'ip')), self.get((server_id, 'port'))
    
    def get_server_id_by_ip_port(self, ip, port):
        with self.lock:
            for server_id, server_info in self.server_info.items():
                if server_info['ip'] == ip and server_info['port'] == port:
                    return server_id
        return None

    def to_json(self):
        with self.lock:
            return copy.deepcopy(self.server_info)
