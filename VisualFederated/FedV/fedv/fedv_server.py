"""
> 2024/02/01
> yue yijie, jaysonyue@outlook.sg
starting aggregation server
"""
from argparse import ArgumentParser
import importlib
import json
import flwr as fl
import sys
#from flwr.server.client_mamager import SimpleClientManager

from fedv import get_data_dir
from fedv.fl_utils.data_loader import job_download
from db.task_dao import TaskDao
from utils.consts import TaskStatus
# from fedv.fl_utils.extension import universal_loader


def main(args):
    try:
        with open(args.config, 'r') as f:
            general_config = json.load(f)

        with open(args.algorithm_config, 'r') as f:
            algorithm_config = json.load(f)

        program = algorithm_config['program']
        architec = algorithm_config['architecture']
        strategy = algorithm_config.get('strategy')
        num_rounds = int(algorithm_config.get('max_iter'))
        download_url = algorithm_config.get('download_url')
        resume = general_config.get('resume', False)
        if resume in ['False', 'false', 'FALSE']:
            resume = False

        module = importlib.import_module(f'fedv.algorithms.{program}.{architec}')
        get_param = getattr(module, 'server_init_params')
        fit_config = getattr(module, 'fit_config')
        metric_agg = getattr(module, 'metric_aggregation')
        data_converter = getattr(module, 'data_converter')
        data_config = getattr(module, 'config_fn')
        data_path = data_converter(job_download(download_url, args.job_id, get_data_dir()))
        data_cfg = data_config(data_path)

        port = args.server_address.split(':')[-1]

        if strategy in ['FedAvg']:
            from fedv.strategy.FedAvg import SecureAggWeighted_Strategy
            strategy_instance = SecureAggWeighted_Strategy(min_fit_clients=int(
                general_config['worker_num']), min_available_clients=int(general_config['worker_num']), initial_parameters=get_param(resume, **data_cfg), on_fit_config_fn=fit_config, fit_metrics_aggregation_fn=metric_agg)
        else:
            raise Exception("Strategy not supported")

        server_config = fl.server.ServerConfig(num_rounds=num_rounds)
        fl.server.start_server(
            server_address=f'[::]:{port}', config=server_config, strategy=strategy_instance)

    except Exception as e:
        TaskDao(args.web_task_id).update_task_status(TaskStatus.ERROR, "联邦聚合服务执行错误")
        sys.stderr.write(str(e))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--algorithm_config', required=True)
    parser.add_argument('--server_address', required=True)
    parser.add_argument('--job_id', required=True)
    parser.add_argument('--web_task_id', required=True)
    args = parser.parse_args()
    
    main(args)
