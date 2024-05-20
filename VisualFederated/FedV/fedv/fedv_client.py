"""
> 2024/02/01
> yueyijie, jaysonyue@outlook.sg
execution training task: federated client
"""

import flwr as fl
import json
import importlib
import os
from argparse import ArgumentParser
import sys

from fedv.security import dh_exchange, asymmetric_encryptor
from fedv import get_data_dir
from fedv.fl_utils.data_loader import job_download
from db.task_dao import TaskDao
from utils.consts import TaskStatus, ComponentName, TaskResultType
from fedv.fl_utils.audit_logger import Auditor


def main(args):
    try:
        with open(args.config, 'r') as f:
            general_config = json.load(f)

        with open(args.algorithm_config, 'r') as f:
            algorithm_config = json.load(f)

        program = algorithm_config.get('program')
        architec = algorithm_config.get('architecture')
        num_rounds = int(algorithm_config.get('max_iter'))
        lr_scaler = float(algorithm_config.get('base_lr'))
        download_url = algorithm_config.get('download_url')
        batch_size = int(algorithm_config.get('batch_size'))
        device = general_config.get('device')

        module = importlib.import_module(
            f'fedv.algorithms.{program}.{architec}')
        fedv_client = getattr(module, 'fedv_client')
        data_converter = getattr(module, 'data_converter')
        data_config = getattr(module, 'config_fn')
        data_dir = job_download(download_url, args.job_id, get_data_dir())
        # redundant code to keep DB as previous
        labelpath = os.path.join(data_dir, "label_list.txt")
        TaskDao(args.web_task_id).save_task_result(
            {"label_path": labelpath}, ComponentName.CLASSIFY if program == 'classification' else ComponentName.DETECTION, TaskResultType.LABEL)
        auditor = Auditor(TaskDao(args.web_task_id).get_flow_id(), args.web_task_id)
        data_path = data_converter(data_dir)
        data_cfg = data_config(data_path)

        ranks = list(range(int(general_config.get('worker_num'))))
        local = int(args.local_idx)

        seeds = dh_exchange(args.keychain_address, local, ranks, args.job_id, auditor)
        mask = asymmetric_encryptor(seeds, local, lr_scaler, num_rounds, auditor)

        client_instance = fedv_client(data=data_path, encrypt_mask=mask, max_iter=num_rounds,
                                      device=device, batch_size=batch_size, web_task_id=args.web_task_id, **data_cfg)
        fl.client.start_numpy_client(
            server_address=args.server_address, client=client_instance)

    except Exception as e:
        TaskDao(args.web_task_id).update_task_status(TaskStatus.ERROR, "联邦学习节点执行错误")
        sys.stderr.write(str(e))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--algorithm_config', required=True)
    parser.add_argument('--server_address', required=True)
    parser.add_argument('--job_id', required=True)
    parser.add_argument('--keychain_address', required=True)
    parser.add_argument('--local_idx', required=True)
    parser.add_argument('--web_task_id', required=True)
    args = parser.parse_args()
    main(args)
