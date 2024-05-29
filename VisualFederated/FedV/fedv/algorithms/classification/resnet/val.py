from .alg_utils import get_validator
import torch
import sys
import os

EPS = 1e-6

def val(data_path, model, device, cfg=None):
    batch = cfg.get('batch_size', 8) if cfg else 8
    validator = get_validator(
        data=data_path, device=device, batch=batch)
    if isinstance(model, str):
        model = torch.load(model)
    metrics = validator(model=model, device=device)
    _pr = None
    _roc = None
    cls_res = None

    _roc = metrics.pop('roc')
    metric_formatted = {k: round(v, 5) for k, v in metrics.items() if k != 'cls_res'}


    try:
        lab_path = os.path.join(data_path, 'label_list.txt')
        id2cls = dict([l.strip().split() for l in open(lab_path, 'r').readlines()])
        _cls_res = metrics.get('cls_res')
        cls_res = {}
        for k, d in _cls_res.items():
            _precision = round(d['tp'] / (d['tp'] + d['fp'] + EPS),5)
            _recall = round(d['tp'] / (d['tp'] + d['fn'] + EPS),5)
            _f1 = round(2 / (1/_precision + 1/_recall),5)
            cls_res[id2cls[str(k)]] = {"precision": _precision, "recall": _recall, "f1": _f1}
    except Exception as e:
        sys.stderr.write(str(e))

    results = {"metrics": metric_formatted,
               "pr": _pr, "roc": _roc, "detail": cls_res}

    return results
