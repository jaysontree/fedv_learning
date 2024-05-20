from .server import server_init_params, fit_config, metric_aggregation
from .client import flClient as fedv_client
from .val import val as val_func
from .infer import predict as infer_func
from fedv.fl_utils.data_processor import _to_yolo as data_converter

def config_fn(data_path):
    return {'data_path': data_path}

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