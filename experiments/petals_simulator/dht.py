import copy
import threading


# TODO: add a reverse lookup table: stage -> server_id (replicas)
# TODO: we could add more utility functions to this class
# NOTE: currently we do not simulate latency in updating and querying the DHT
class DistributedHashTable:
    INFO_TYPES = ['ip', 'port', 'location', 'status', 'stages', 'load']

    def __init__(self):
        self.lock = threading.Lock()
        self.server_info = dict()  # server_id -> (ip, port, location, status, stages, load)

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

    def to_json(self):
        with self.lock:
            return copy.deepcopy(self.server_info)
