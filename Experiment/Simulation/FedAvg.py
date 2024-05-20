from typing import List, Tuple
from collections import OrderedDict
import numpy as np
from functools import reduce
import flwr
from flwr.common import Parameters, ndarrays_to_parameters

class SecureAggWeighted_Strategy(flwr.server.strategy.FedAvg):
    def aggregate_fit(self, server_round: int, results: List[Tuple[flwr.server.client_proxy.ClientProxy, flwr.common.FitRes]], failures: List[Tuple[flwr.server.client_proxy.ClientProxy, flwr.common.FitRes, BaseException,]]):
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}
        scale_f = 1. / sum([fitres.num_examples for _, fitres in results])
        params = [x for x in flwr.common.parameters_to_ndarrays(
            results[0][1].parameters)]
        for i, (_, fitres) in enumerate(results[1:]):
            res = (x for x in flwr.common.parameters_to_ndarrays(fitres.parameters))
            params = [reduce(np.add, layer_updates)
                      for layer_updates in zip(params, res)]

        weights = [x * scale_f for x in params]
        weights_agg = ndarrays_to_parameters(weights)
        return weights_agg, {}
