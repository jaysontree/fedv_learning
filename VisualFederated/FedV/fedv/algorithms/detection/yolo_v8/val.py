from .alg_utils import get_validator
import torch
import sys


def val(data_path, model_path, device, cfg=None):
    batch = cfg.get('batch_size', 8) if cfg else 8
    validator = get_validator(
        data=data_path, device=device, batch=batch)
    if isinstance(model_path, str):
        model = torch.load(model_path)
    else:  # to support direct call from clients.
        model = model_path
    metrics = validator(model=model)
    metric_formatted = {}
    for k, v in metrics.items():
        if k == 'fitness':
            continue
        if k.startswith("metrcis"):
            k = k[8:]
        metric_formatted[k] = v

    apindex = validator.metrics.ap_class_index
    names = validator.metrics.names

    _pr = None
    _roc = None
    _cls_res = None
    try:
        _pr_curve = validator.metrics.curves_results[0]
        aps = _pr_curve[1]
        _pr_x_axis = _pr_curve[0]
        overall_ap = aps.mean(0)
        assert aps.shape[0] == len(apindex)
        _pr_y_axis = {}
        _cls_res = {}
        for i, idx in enumerate(apindex):
            _pr_y_axis[names[idx]] = aps[i]
            _cls_res[names[idx]] = {"precision": validator.metrics.class_result(i)[0], "recall": validator.metrics.class_result(
                i)[1], "mAP50": validator.metrics.class_result(i)[2], "mAP50-95": validator.metrics.class_result(i)[3]}
        _cls_res.update({"all_classes": {"precision": validator.metrics.mean_results()[0], "recall": validator.metrics.mean_results()[
                        1], "mAP50": validator.metrics.mean_results()[2], "mAP50-95": validator.metrics.mean_results()[3]}})
        _pr_y_axis.update({"all_classes": overall_ap})
        _pr = {'x_axis': _pr_x_axis, 'y_axis': _pr_y_axis}

    except Exception as e:
        sys.stderr.write(str(e))

    results = {"metrics": metric_formatted,
               "pr": _pr, "roc": _roc, "detail": _cls_res}

    return results
