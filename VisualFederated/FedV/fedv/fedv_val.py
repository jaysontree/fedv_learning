"""
> 2024/02/01
> yueyijie, jaysonyue@outlook.sg
execution validation task
"""
import json
import importlib
from argparse import ArgumentParser
from pathlib import Path
import os
import sys

from fedv.fl_utils.data_loader import job_download
from fedv import __logs_dir__
from db.task_dao import TaskDao
from utils.consts import TaskStatus, ComponentName, TaskResultType
from fedv.fl_utils.audit_logger import Auditor

def main(args):
    try:
        DAO = TaskDao(args.web_task_id)
        dm = DAO.get_task_result(TaskResultType.VAL)
        if dm:
            dm_results = json.loads(dm.result)
        else:
            dm_results = {}
        with open(args.config, 'r') as f:
            general_config = json.load(f)

        with open(args.algorithm_config, 'r') as f:
            algorithm_config = json.load(f)

        program = algorithm_config.get('program')
        architec = algorithm_config.get('architecture')
        batch_size = int(algorithm_config.get('batch_size'))
        device = general_config.get('device')

        download_url = general_config.get('download_url')
        input_dir = Path(__logs_dir__).joinpath(f"jobs/{args.job_id}/val/input")
        val_session_id = general_config.get('val_session_id')

        module = importlib.import_module(f'fedv.algorithms.{program}.{architec}')
        data_converter = getattr(module, 'data_converter')

        val_dir = data_converter(job_download(download_url, val_session_id, input_dir))
        #local_trainer_idx = general_config.get('local_trainer_indexs')[0] # 现在flow传过来的有问题、这边做下容错，读原任务的信息、或者路径搜索
        # weights = Path(__logs_dir__).joinpath(f"jobs/{args.oot_job_id}/trainer_{local_trainer_idx}/model.pt")
        train_dirs = [d for d in os.listdir(Path(__logs_dir__).joinpath(f"jobs/{args.oot_job_id}")) if d.startswith('trainer_')] # 暂时方案
        weights = Path(__logs_dir__).joinpath(f"jobs/{args.oot_job_id}/{train_dirs[0]}/model.pt") # 暂时方案

        valfn = getattr(module, 'val_func')
        cfg = {'batch_size': batch_size}

        dm_results.update({'status': "running"})   
        componentname =  ComponentName.DETECTION if program == "detection" else ComponentName.CLASSIFY
        DAO.save_task_result(
            dm_results, componentname, type=TaskResultType.VAL)
        metrics = valfn(val_dir, str(weights), device, cfg)
        dm_results.update({'status': 'finish', 'results': metrics})
        DAO.save_task_result(
            dm_results, componentname, type=TaskResultType.VAL)
        auditor = Auditor(TaskDao(args.web_task_id).get_flow_id(), args.web_task_id)
        auditor.info("No encryption information provided. Computation is handled locally during validation process")
        TaskDao(args.web_task_id).update_task_status(TaskStatus.SUCCESS)

    except Exception as e:
        TaskDao(args.web_task_id).update_task_status(TaskStatus.ERROR, "打分验证流程运行错误")
        dm_results.update({'status': "error", 'message': str(e)})
        componentname =  ComponentName.DETECTION if program == "detection" else ComponentName.CLASSIFY
        DAO.save_task_result(dm_results, componentname, type=TaskResultType.VAL)
        sys.stderr.write(str(e))

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--algorithm_config', required=True)
    parser.add_argument('--job_id', required=True)
    parser.add_argument('--web_task_id', required=True)
    parser.add_argument('--oot_job_id', required=True)
    args = parser.parse_args()
    main(args)