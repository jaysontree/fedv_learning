from .alg_utils import get_predictor
import torch
import os
from typing import Tuple 
import numpy as np

def predict(data_path, output_dir, model_path, device, cfg=None):
    try:
        predictor = get_predictor(device=device, batch=cfg.get('batch_size', 32))
        label_path = cfg.get('label_path')
        id_2_cls = dict([line.strip().split() for line in open(label_path,'r').readlines()])
        model = torch.load(model_path)
        raw_result = predictor(source=data_path, model=model)
        processed_result = _process(raw_result, id_2_cls)
    except Exception as e:
        print(e)
        raise Exception
    return processed_result

def softmax(arr: Tuple[np.ndarray, list]) -> np.ndarray:
    return np.exp(arr) / np.sum(np.exp(arr))

def _process(res,id2cls):
    new_res = []
    for r in res:
        image_name = r[0]
        infer_res = r[1]
        infer_probs = []
        scores = softmax(infer_res)
        for idx, score in enumerate(scores):
            infer_probs.append(
                {
                    "class_id": idx,
                    "class_name": id2cls[str(idx)],
                    "prob": round(float(score), 6)
                }
            )
        new_res.append({"image": image_name, "infer_probs": infer_probs})
    return new_res