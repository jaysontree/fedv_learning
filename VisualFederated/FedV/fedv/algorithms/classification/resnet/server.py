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

from .alg_utils import get_weights

def server_init_params(resume, nc, suffix):
    if resume and Path.exists('model.pt'):
        return ndarrays_to_parameters([v for _, v in torch.load('model.pt').state_dict().items()])
    else:
        return ndarrays_to_parameters([v for _, v in get_weights(nc, suffix).state_dict().items()])

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
    class SecureAggWeighted_Strategy(flwr.server.strategy.FedAvg):
        def aggregate_fit(self, server_round: int, results: List[Tuple[flwr.server.client_proxy.ClientProxy, flwr.common.FitRes]], failures: List[Tuple[flwr.server.client_proxy.ClientProxy, flwr.common.FitRes, BaseException,]]):
            if not results:
                return None, {}
            if not self.accept_failures and failures:
                return None, {}
            scale_f = 1. / sum([fitres.num_examples for _, fitres in results])
            params = [x for x in flwr.common.parameters_to_ndarrays(results[0][1].parameters)]
            for i, (_, fitres) in enumerate(results[1:]):
                res = (x for x in flwr.common.parameters_to_ndarrays(fitres.parameters))
                params = [reduce(np.add, layer_updates) for layer_updates in zip(params, res)]

            weights = [ x * scale_f for x in params]
            weights_agg = ndarrays_to_parameters(weights)
            return weights_agg, {}
    strategy = SecureAggWeighted_Strategy(min_fit_clients=2, initial_parameters=server_init_params(resume=False), on_fit_config_fn=fit_config)
    server_config = flwr.server.ServerConfig(num_rounds=100)
    flwr.server.start_server(server_address='[::]:31101', config=server_config, strategy=strategy)
