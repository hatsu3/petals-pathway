import uuid


# TODO: make this a dataclass
# A request to run a task in the multi-task model
# initialized with the IP address of the client and the task ID
class InferRequest:
    def __init__(self, request_id: uuid.UUID, client_ip: str, client_port: int, task_id: int):
        self.request_id = request_id
        self.client_ip = client_ip
        self.client_port = client_port
        self.task_id = task_id
        
        # The index of the next stage to execute, initialized to 0
        self.next_stage_idx = 0

    def update(self) -> "InferRequest":
        self.next_stage_idx += 1
        return self
    
    def to_json(self):
        return {
            "request_id": str(self.request_id),
            "client_ip": self.client_ip,
            "client_port": self.client_port,
            "task_id": self.task_id,
            "next_stage_idx": self.next_stage_idx
        }
    
    @classmethod
    def from_json(cls, json):
        request_id = uuid.UUID(json["request_id"])
        request = cls(request_id, json["client_ip"], json["client_port"], json["task_id"])
        request.next_stage_idx = json["next_stage_idx"]
        return request


class InferResponse:
    def __init__(self, request: InferRequest, result: float):
        self.request = request
        self.result = result

    @property
    def request_id(self):
        return self.request.request_id

    def to_json(self):
        return {
            "request": self.request.to_json(),
            "result": self.result
        }
    
    @classmethod
    def from_json(cls, json):
        request = InferRequest.from_json(json["request"])
        return cls(request, json["result"])
