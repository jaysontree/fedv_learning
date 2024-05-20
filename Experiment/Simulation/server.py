from typing import Optional, List, Tuple, Dict
from collections import OrderedDict
import flwr
import numpy as np
import torch
from pathlib import Path
from flwr.server.client_manager import SimpleClientManager
from flwr.common import Parameters, ndarrays_to_parameters
from flwr.server import Server
from functools import reduce

from alg_utils import get_weights
from FedAvg import SecureAggWeighted_Strategy

def server_init_params(resume, data_path):
    if resume and Path.exists('model.pt'):
        return ndarrays_to_parameters([v for _, v in torch.load('model.pt').state_dict().items()])
    else:
        return ndarrays_to_parameters([v for _, v in get_weights(data=data_path, imgsz=640, device='cpu').state_dict().items()])

def fit_config(server_round: int):
    config = {'current_round': server_round}
    return config

def metric_aggregation(metrics: List[Tuple[int, Dict]]) -> Dict:
    keys = metrics[0][1].keys()
    sample_sum = sum(num_examples for num_examples, _ in metrics)
    overall = {}
    for key in keys:
        overall[key] = sum(num_examples * m[key] for num_examples, m in metrics) / sample_sum
    return overall


if __name__=='__main__':
    strategy = SecureAggWeighted_Strategy(min_fit_clients=3, min_available_clients=3, initial_parameters=server_init_params(resume=False, data_path="data1.yaml"), on_fit_config_fn=fit_config, fit_metrics_aggregation_fn=metric_aggregation)
    server_config = flwr.server.ServerConfig(num_rounds=100)
    flwr.server.start_server(server_address='[::]:40101', config=server_config, strategy=strategy)
