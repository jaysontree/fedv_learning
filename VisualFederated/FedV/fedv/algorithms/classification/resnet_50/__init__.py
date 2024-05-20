from fedv.algorithms.classification.resnet import data_converter, server_init_params, fit_config, metric_aggregation
from fedv.algorithms.classification.resnet import flClient as fedv_client
from fedv.algorithms.classification.resnet.val import val as val_func
from fedv.algorithms.classification.resnet.infer import predict as infer_func

SUFFIX='50'

def config_fn(data_path):
    import os
    label_f = os.path.join(data_path,'label_list.txt')
    nc = len([x for x in open(label_f,'r').readlines() if len(x.strip())>2])
    return {'nc': nc, 'suffix': SUFFIX}

__all__ = [
    'server_init_params',
    'fedv_client',
    'fit_config',
    'metric_aggregation',
    'data_converter',
    'val_func',
    'infer_func',
    'config_fn'
]