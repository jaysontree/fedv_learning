from utils.logger import set_logger, Logger

class Auditor(Logger):
    def __init__(self, flow_id, task_id):
        self.identificator=f"audit-{flow_id}-{task_id}"
        set_logger(self.identificator)

