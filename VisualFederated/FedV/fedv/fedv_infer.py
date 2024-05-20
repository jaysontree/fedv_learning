"""
> 2024/02/01
> yueyijie, jaysonyue@outlook.sg
execution inference task
"""
import json
import importlib
from argparse import ArgumentParser
from pathlib import Path
import sys

from fedv.fl_utils.data_loader import job_download, extractImages
from fedv import __logs_dir__
from db.task_dao import TaskDao
from utils.consts import TaskStatus, ComponentName, TaskResultType

def main(args):
    try:
        DAO = TaskDao(args.web_task_id)
        dm = DAO.get_task_result(TaskResultType.INFER)
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
        input_dir = Path(__logs_dir__).joinpath(f"jobs/{args.job_id}/infer/input")
        infer_session_id = general_config.get('infer_session_id')

        infer_dir = job_download(download_url, infer_session_id, input_dir)
        extractImages(infer_dir)
        output_dir = Path(__logs_dir__).joinpath(f"jobs/{args.job_id}/infer/output/{Path(infer_dir).name}")

        module = importlib.import_module(f'fedv.algorithms.{program}.{architec}')
        local_trainer_idx = general_config.get('local_trainer_indexs')[0]
        weights = Path(__logs_dir__).joinpath(f"jobs/{args.job_id}/trainer_{local_trainer_idx}/model.pt")

        inferfn = getattr(module, 'infer_func')

        label_m = DAO.get_task_result(TaskResultType.LABEL)
        if label_m:
            label_path = json.loads(label_m.result)['label_path']
        cfg = {'batch_size': batch_size, 'label_path': label_path}

        dm_results.update({'status': "running"})   
        componentname =  ComponentName.DETECTION if program == "detection" else ComponentName.CLASSIFY
        DAO.save_task_result(
            dm_results, componentname, type=TaskResultType.INFER)
        results = inferfn(infer_dir, output_dir, weights, device, cfg)
        dm_results.update({'status': 'finish', 'result': results})
        DAO.save_task_result(
            dm_results, componentname, type=TaskResultType.INFER)

    except Exception as e:
        # TaskDao(args.web_task_id).update_task_status(TaskStatus.ERROR, str(e)) # 预测不会创建Task
        dm_results.update({'status': "error", 'message': str(e)})
        componentname =  ComponentName.DETECTION if program == "detection" else ComponentName.CLASSIFY
        DAO.save_task_result(dm_results, componentname, type=TaskResultType.INFER)
        sys.stderr.write(str(e))

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--algorithm_config', required=True)
    parser.add_argument('--job_id', required=True)
    parser.add_argument('--web_task_id', required=True)
    args = parser.parse_args()
    main(args)