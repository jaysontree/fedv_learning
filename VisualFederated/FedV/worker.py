"""
> 2024/02/02
> Yue Yijie, jaysonyue@outlook.sg
Federated learning execution(internal) service
"""
import asyncio
from typing import Optional
import grpc
import socket
from pathlib import Path

from fedv import __logs_dir__
import fedv.protobuf.worker_pb2 as worker_pb2
import fedv.protobuf.worker_pb2_grpc as worker_pb2_grpc
from utils.logger import Logger
from fedv.fl_utils.executor import ProcessExecutor
from utils.consts import TaskStatus
from db.task_dao import TaskDao

class Worker(Logger):

    def __init__(self, host: str, port: int, max_tasks: int, port_start: int, port_end: int, data_dir: str = None):
        self._host = "[::]" if host is None else host
        self._port = port
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._semaphore = asyncio.Semaphore(max_tasks)
        self._max_tasks = max_tasks
        self._port_start = port_start
        self._port_end = port_end
        assert self._port_end > self._port_start
        self._cur_port = port_start
        self._server: Optional[grpc.aio.Server] = None
        self._data_dir = data_dir

    async def start(self):
        """
        start worker service
        Returns:

        """
        self.info(f"starting cluster manager at port: {self._port}")
        self._server = grpc.aio.server(
            options=[
                ("grpc.max_send_message_length", 512 * 1024 * 1024),
                ("grpc.max_receive_message_length", 512 * 1024 * 1024),
            ],
        )
        worker_pb2_grpc.add_FedWorkerServicer_to_server(self, self._server)
        self._server.add_insecure_port(f"[::]:{self._port}")
        await self._server.start()
        self.info(f"worker started at port: {self._port}")
        asyncio.create_task(self._co_task_execute_loop())

    async def stop(self):
        """
        stop worker service:
        Returns:

        """
        await self._server.stop(1)

    async def TaskResourceRequire(self, request: worker_pb2.Resource.REQ, context) -> worker_pb2.Resource.REP:
        port_candidates = []
        for i in range(1, self._port_end - self._port_start + 1):
            port_id = (self._cur_port + i) % (self._port_end -
                                              self._port_start) + self._port_start
            self.debug(port_id)
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port_id))
                    s.close()
                    port_candidates.append(f"{self._host}:{port_id}")
                if len(port_candidates) >= request.num_endpoints:
                    self.debug('success')
                    response = worker_pb2.Resource.REP(
                        status=worker_pb2.Resource.SUCCESS, endpoints=port_candidates)
                    self._cur_port = port_id
                    return response
            except OSError as e:
                self.debug(e)
        response = worker_pb2.Resource.REP(status=worker_pb2.Resource.FAILED)
        return response


    async def TaskSubmit(self, request: worker_pb2.TaskSubmit.REQ, context) -> worker_pb2.TaskSubmit.REP:
        try:
            task = request.task_submit
            await self._task_queue.put(task)
            return worker_pb2.TaskSubmit.REP(status=worker_pb2.TaskSubmit.SUCCESS)
        except Exception as e:
            return worker_pb2.TaskSubmit.REP(status=worker_pb2.TaskSubmit.FAILED)

    async def _task_exec_coroutine(self, _task):
        try:
            self.info(
                f"start to exec task, job_id={_task.job_id}, task_id={_task.task_id}, task_type={_task.task_type}"
            )
            executor = ProcessExecutor(
                Path(__logs_dir__).joinpath(
                    f"jobs/{_task.job_id}/{_task.task_id}"),
                data_dir=self._data_dir,
            )
            response = await executor.execute(_task.task)
            self.info(
                f"finish exec task, job_id={_task.job_id}, task_id={_task.task_id}"
            )

            self.trace(f"update task status")

        except Exception as e:
            self.exception(e)
            TaskDao(_task.web_task_id).update_task_status(
                TaskStatus.ERROR, str(e))
        finally:
            self._semaphore.release()
            self.trace_lazy(
                f"semaphore released, current: {{current}}",
                current=lambda: self._semaphore,
            )

    async def _co_task_execute_loop(self):

        # noinspection PyUnusedLocal

        while True:
            self.trace(f"acquiring semaphore")
            await self._semaphore.acquire()
            self.trace(f"acquired semaphore")
            self.trace(f"get from task queue")
            ready_task = await self._task_queue.get()
            self.trace_lazy(f"got {{task}} from task queue",
                            task=lambda: ready_task)
            asyncio.create_task(self._task_exec_coroutine(ready_task))
            self.trace(f"asyncio task created to exec task")
