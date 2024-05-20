from .alg_utils import get_predictor
import torch
import os

def predict(data_path, output_dir, model_path, device, cfg=None):
    try:
        predictor = get_predictor(device=device, batch=cfg.get('batch_size'))
        model = torch.load(model_path)
        raw_result = predictor(source=data_path, model=model)
        processed_result = _process(raw_result)
    except Exception as e:
        print(e)
        raise Exception
    return processed_result

def _process(res):
    new_res = []
    for r in res:
        image_name = os.path.basename(r.path)
        cat_2_name = r.names
        bbox_results = []
        for _i, _d in enumerate(r.boxes.data.cpu().numpy().tolist()):
            category_id = int(_d[5])
            category_name = cat_2_name.get(category_id)
            bbox = _d[:4]
            score = _d[4]
            bbox_result = {
                "image_id": 0,
                "category_id": category_id,
                "bbox": bbox,
                "score": score,
                "category_name": category_name,
            }
            bbox_results.append(bbox_result)
        new_res.append({"image": image_name, "bbox_results": bbox_results})
    return new_res