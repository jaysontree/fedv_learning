"""
> 2024/02/01
> yueyijie, jaysonyue@outlook.sg
modified instance from ultralytics official model
"""
import math
import os
import torch
from ultralytics import YOLO
from ultralytics.cfg import TASK2DATA, get_cfg
from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT, checks, yaml_load, callbacks, LOGGER, ops
from ultralytics.utils.torch_utils import select_device
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator, DetectionPredictor

from fedv import get_base_dir

class FedVTrainer(DetectionTrainer):
    """
    disable features which are not needed/supported in federated trainning/swarm trainning.
    disable inplace data parallelism
    """
    def __init__(self, cfg = DEFAULT_CFG, overrides=None, _callbacks=None):
        self.args = get_cfg(cfg, overrides)
        self.check_resume(overrides)
        self.device = select_device(self.args.device, self.args.batch)
        self.validator = None #
        self.metrics = None #
        self.plots = {} #

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs
        self.start_epoch = 0

        self.args.workers = 0 # force not to use share memory for data loader. container only has 64MB shared memeory by default
        self.model = checks.check_model_file_from_stem(self.args.model)
        self.data = check_det_dataset(self.args.data)

        self.trainset, self.testset = self.get_dataset(self.data)
        self.lf = None
        self.scheduler = None
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ['Loss']

        self.callbacks =  callbacks.get_default_callbacks()

    def _get_net_params(self):
        self.setup_model()
        return self.model

    def _get_net_ckpt(self):
        ckpt = self.setup_model()
        return ckpt
        
    def _setup_train(self):
        ckpt = self._get_net_ckpt()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # freeze layers
        freeze_list = self.args.freeze if isinstance(self.args.freeze, list) else range(self.args.freeze) if isinstance(self.args.freeze, int) else []
        always_freeze_names = ['.dfl']
        freeze_layer_names = [f'model.{x}.' for x in freeze_list] + always_freeze_names
        for k, v in self.model.named_parameters():
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f'Freezing layer "{k}"')
                v.requires_grad=False
            elif not v.requires_grad:
                LOGGER.info(f'setting "requires_grad=True" for frozen layer {k}')
                v.requires_grad=True

        # disable AMP
        self.amp = False

        gs = max(int(self.model.stride.max() if hasattr(self.model, 'stride') else 32), 32)
        self.args.imgsz = checks.check_imgsz(self.args.imgsz, stride=gs, max_dim=1)

        # init dataloader
        self.train_loader = self.get_dataloader(self.trainset, batch_size = self.batch_size, rank=-1, mode='train')
        self.test_loader = self.get_dataloader(self.testset, batch_size = self.batch_size*2, rank=-1, mode='val')

        self.ema = None # TODO, is it possible to use exponential move average on the server side?

        # init optimizer
        iterations = math.ceil(len(self.train_loader.dataset) / self.batch_size) * self.epochs
        self.optimizer = self.build_optimizer(model=self.model, name=self.args.optimizer, lr=self.args.lr0, momentum=self.args.momentum, decay=self.args.weight_decay, iterations=iterations)

        # init scheduler
        self._setup_scheduler()

        # init validator TODO
        self.save_dir=None
        self.validator = self.get_validator()
        #metrics_keys = self.validator.metrics.keys + self.label_loss_items(prefix='val')


def check_base_model():
    if not os.path.exists('yolov8n.pt'):
        base_model_path = os.path.join(get_base_dir(), 'algorithms/detection/yolo_v8/yolov8n.pt')
        os.symlink(base_model_path, 'yolov8n.pt')

def get_weights(**kwargs):
    check_base_model()
    model = YOLO('yolov8n.pt')
    overrides = model.overrides
    custom = {'data': DEFAULT_CFG_DICT['data'] or TASK2DATA[model.task]}
    args = {**overrides, **custom, **kwargs, 'mode': 'train', 'plots': False, 'save_json': False, 'save_txt': False}
    model.trainer = FedVTrainer(overrides=args)
    return model.trainer._get_net_params()

def get_trainer(**kwargs):
    check_base_model()
    model = YOLO('yolov8n.pt')
    overrides = model.overrides
    custom = {'data': DEFAULT_CFG_DICT['data'] or TASK2DATA[model.task]}
    args = {**overrides, **custom, **kwargs, 'mode': 'train', 'plots': False, 'save_json': False, 'save_txt': False}
    model.trainer = FedVTrainer(overrides=args)
    model.trainer._setup_train()
    return model.trainer

def get_validator(**kwargs):
    check_base_model()
    model = YOLO('yolov8n.pt')
    overrides = model.overrides
    custom = {'data': DEFAULT_CFG_DICT['data'] or TASK2DATA[model.task]}
    args = {**overrides, **custom, **kwargs, 'mode': 'val', 'plots': True, 'save_json': False, 'save_txt': False}
    validator = DetectionValidator(args=args)
    return validator

def get_predictor(**kwargs):
    check_base_model()
    model = YOLO('yolov8n.pt')
    overrides = model.overrides
    custom = {'data': DEFAULT_CFG_DICT['data'] or TASK2DATA[model.task]}
    args = {**overrides, **custom, **kwargs, 'mode': 'predict', 'show': False, 'verbose': False, 'visualize': False, 'save': False, 'save_txt': False, 'save_crop': False}    
    predictor = DetectionPredictor(overrides=args)
    return predictor
