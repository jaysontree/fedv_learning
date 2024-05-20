import sys
from typing import List
import json
from pathlib import Path

from fedv import __logs_dir__
from fedv.base.abs.job import Job
import fedv.protobuf.worker_pb2 as worker_pb2


class FedVJob(Job):

    @classmethod
    def load(cls, job_id, task_id, role, member_id, config, algorithm_config, callback_url=None):
        job = FedVJob(job_id=job_id, task_id=task_id,
                      role=role, member_id=member_id)
        job._init_job(config, algorithm_config, callback_url)
        return job

    def __init__(self, job_id, task_id, role, member_id):
        self.job_id = job_id
        self._role = role
        self._member_id = member_id
        self._web_task_id = task_id

    def _init_job(self, config, algorithm_config, callback_url=None):
        self._worker_num = config["worker_num"]
        self._local_worker_num = config["local_worker_num"]
        self._local_trainer_indexs = config["local_trainer_indexs"]
        self._program = algorithm_config["program"]
        self._config_string = json.dumps(config)
        self._algorithm_config = json.dumps(algorithm_config)
        self._server_endpoint = config.get("server_endpoint", None)
        self._aggregator_endpoint = config.get("aggregator_endpoint", None)
        self._oot_job_id = config.get("oot_job_id", None)
        self._callback_url = callback_url

    @property
    def resource_required(self):
        return worker_pb2.Resource.REQ(num_endpoints=2)

    def set_required_resource(self, response):
        self._server_endpoint = response.endpoints[0]
        self._aggregator_endpoint = response.endpoints[1]

    @property
    def val_path(self):
        return Path(__logs_dir__).joinpath(f"jobs/{self.job_id}/val")

    @property
    def infer_path(self):
        return Path(__logs_dir__).joinpath(f"jobs/{self.job_id}/infer")

    def generate_trainer_tasks(self) -> List[worker_pb2.Task]:
        tasks = []
        for i, v in enumerate(self._local_trainer_indexs):
            tasks.append(self._generate_trainer_task_pb(v))
        return tasks

    def generate_aggregator_tasks(self) -> List[worker_pb2.Task]:
        return [self._generate_key_exchanger_task_pb(), self._generate_aggregator_task_pb()]

    def _generate_trainer_task_pb(self, idx):

        execution_path = Path(__logs_dir__).joinpath(f'jobs/{self.job_id}/trainer_{idx}')
        if not Path.exists(execution_path):
            execution_path.mkdir(parents=True, exist_ok=True)
        with open(execution_path.joinpath('config.json'), 'w') as f:
            f.write(self._config_string)
        with open(execution_path.joinpath('algorithm_config.json'), 'w') as f:
            f.write(self._algorithm_config)

        task_pb = worker_pb2.Task(
            job_id=self.job_id,
            task_id=f"trainer_{idx}",
            web_task_id=self._web_task_id,
            task_type='fedv_trainer'
        )
        executable = sys.executable
        trainer_task = ' '.join(
            [
                f"{executable} -m fedv.fedv_client",
                f"--job_id {self.job_id}",
                f"--local_idx {idx}",
                f"--web_task_id {self._web_task_id}",
                f"--server_address {self._aggregator_endpoint}",
                f"--keychain_address {self._server_endpoint}",
                f"--config config.json",
                f"--algorithm_config algorithm_config.json",
                f">stdout 2>stderr"
            ]
        )
        task_pb.task = trainer_task
        return task_pb

    def _generate_aggregator_task_pb(self):

        execution_path = Path(__logs_dir__).joinpath(f'jobs/{self.job_id}/aggregator')
        if not Path.exists(execution_path):
            execution_path.mkdir(parents=True, exist_ok=True)
        with open(execution_path.joinpath('config.json'), 'w') as f:
            f.write(self._config_string)
        with open(execution_path.joinpath('algorithm_config.json'), 'w') as f:
            f.write(self._algorithm_config)

        task_pb = worker_pb2.Task(
            job_id=self.job_id,
            web_task_id=self._web_task_id,
            task_id="aggregator",
            task_type="fedv_aggregator"
        )

        executable = sys.executable
        aggregator_task = " ".join(
            [
                f"{executable} -m fedv.fedv_server",
                f"--job_id {self.job_id}",
                f"--web_task_id {self._web_task_id}",
                f"--server_address {self._aggregator_endpoint}",
                f"--config config.json",
                f"--algorithm_config algorithm_config.json",
                f">stdout 2>stderr"
            ]
        )
        task_pb.task = aggregator_task
        return task_pb

    def _generate_key_exchanger_task_pb(self):
        
        execution_path = Path(__logs_dir__).joinpath(f'jobs/{self.job_id}/security')
        if not Path.exists(execution_path):
            execution_path.mkdir(parents=True, exist_ok=True)
        with open(execution_path.joinpath('config.json'), 'w') as f:
            f.write(self._config_string)
        with open(execution_path.joinpath('algorithm_config.json'), 'w') as f:
            f.write(self._algorithm_config)

        task_pb = worker_pb2.Task(
            job_id=self.job_id,
            web_task_id=self._web_task_id,
            task_id="security",
            task_type="fedv_security"
        )
        executable = sys.executable
        key_exchange_task = ' '.join(
            [
                f"{executable} -m fedv.security.exchange_provider",
                f"--job_id {self.job_id}",
                #f"--web_task_id {self._web_task_id}",
                f"--addr {self._server_endpoint}",
                #f"--config config.json",
                f">stdout 2>stderr",
            ]
        )
        task_pb.task = key_exchange_task
        return task_pb

    def generate_val_task(self):
        execution_path = Path(__logs_dir__).joinpath(f"jobs/{self.job_id}/val")
        if not Path.exists(execution_path):
            execution_path.mkdir(parents=True, exist_ok=True)
        with open(execution_path.joinpath('config.json'), 'w') as f:
            f.write(self._config_string)
        with open(execution_path.joinpath('algorithm_config.json'), 'w') as f:
            f.write(self._algorithm_config)

        task_pb = worker_pb2.Task(
            job_id=self.job_id,
            web_task_id=self._web_task_id,
            task_id="val",
            task_type="fedv_val"
        )

        executable = sys.executable
        val_task = ' '.join(
            [
                f"{executable} -m fedv.fedv_val",
                f"--job_id {self.job_id}",
                f"--config config.json",
                f"--algorithm_config algorithm_config.json",
                f"--web_task_id {self._web_task_id}",
                f"--oot_job_id {self._oot_job_id}",
                f">stdout 2>stderr"
            ]
        )
        task_pb.task = val_task
        return [task_pb]

    def generate_infer_task(self):
        execution_path = Path(__logs_dir__).joinpath(f"jobs/{self.job_id}/infer")
        if not Path.exists(execution_path):
            execution_path.mkdir(parents=True, exist_ok=True)
        with open(execution_path.joinpath('config.json'), 'w') as f:
            f.write(self._config_string)
        with open(execution_path.joinpath('algorithm_config.json'),'w') as f:
            f.write(self._algorithm_config)

        task_pb = worker_pb2.Task(
            job_id=self.job_id,
            web_task_id=self._web_task_id,
            task_id="infer",
            task_type="fedv_infer"
        )

        executable = sys.executable
        infer_task = ' '.join(
            [
                f"{executable} -m fedv.fedv_infer",
                f"--job_id {self.job_id}",
                f"--config config.json",
                f"--algorithm_config algorithm_config.json",
                f"--web_task_id {self._web_task_id}"
                f">stdout 2>stderr"
            ]
        )
        task_pb.task = infer_task
        return [task_pb]