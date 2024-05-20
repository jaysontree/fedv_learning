"""
> 2024/02/01
> yueyijie, jaysonyue@outlook.sg
dependents modified from orginal ResNet porject with torch implementation
"""
import torch
import torchvision
from torchvision import transforms
import os
import PIL
import numpy as np

from .folder import ImageFolder

from fedv import get_base_dir

def get_weights(nc, suffix='18'):
    assert suffix in ['18', '34', '50', '101', '152']
    archit = getattr(torchvision.models, f"resnet{suffix}")
    pretrained_path = os.path.join(get_base_dir(), f"algorithms/classification/resnet_{suffix}/resnet{suffix}.pth")
    if os.path.exists(pretrained_path):
        weights = torch.load(pretrained_path)
        model = archit()
        model.load_state_dict(weights)
        # overwrite last fc layer
        model.fc = torch.nn.Linear(model.fc.in_features, nc)
    else:
        model = archit(num_classes=nc)
    return model

def get_model(nc, suffix='18'):
    archit = getattr(torchvision.models, f'resnet{suffix}')
    return archit(num_classes=nc)

def set_up_optimizer(model, lr, decay):
    # remove weight decay for norm layer compared to original ResNet introduced by Kaiming He
    param_groups = []
    norm_params = []
    other_params = []

    def serach_for_bn(module, prefix):
        for name, pa in module.named_parameters(recurse=False):
            if not pa.requires_grad:
                continue
            if isinstance(pa, torch.nn.modules.batchnorm._BatchNorm):
                norm_params.append(pa)
            else:
                other_params.append(pa)
        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            serach_for_bn(child_module, prefix=child_prefix)

    serach_for_bn(model, prefix='')
    if norm_params:
        param_groups.append(
            {"params": norm_params, "weight_decay": 0.0}) 
    if other_params:
        param_groups.append(
            {"params": other_params, "weight_decay": decay})  

    optimizer = torch.optim.SGD(param_groups, lr=lr, weight_decay=decay)
    return optimizer


def load_dataset(path: str, batch_size: int):
    train_path = os.path.join(path, 'train')
    val_path = os.path.join(path, 'val')
    if os.path.exists(train_path):
        train_dataset = ImageFolder(train_path, transform=transforms.Compose([transforms.Resize(
            (224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]),allow_empty=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size, shuffle="True", num_workers=0)
    else:
        train_loader = None
    if os.path.exists(val_path):
        val_dataset = ImageFolder(val_path, transform=transforms.Compose([transforms.Resize(
            (224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]), allow_empty=True)
        test_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size, shuffle="False", num_workers=0)
    else:
        test_loader = None
    return train_loader, test_loader


class resnetTrainer:
    def __init__(self, data_path, nc, suffix='18', **kwargs):
        self.batch_size = kwargs.get('batch', 64)
        self.epochs = kwargs.get('epochs', 100)
        self.model = get_model(nc, suffix)
        self.train_loader, self.test_loader = load_dataset(data_path, self.batch_size)
        self.optimizer = set_up_optimizer(self.model, lr=.1, decay=1e-4)
        if self.epochs > 40:
            milestone = [round(self.epochs/2), round(self.epochs*3/4), self.epochs - 10, self.epochs - 1]
        else:
            milestone = [round(self.epochs/2)]
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestone, gamma=0.1)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.validator = resnetValidator(self.test_loader)

class resnetValidator:
    def __init__(self, test_loader):
        self.test_loader = test_loader
        self.metrics = {}

    def __call__(self, model, device='cpu'):
        with torch.no_grad():
            model.eval().to(device)
            total = 0.
            correct = 0.
            class_res = {}

            for inp, tgt in self.test_loader:
                inp = inp.to(device)
                out = model(inp)
                _out = torch.softmax(out, 1)
                confs = torch.max(_out, 1)
                predicts = torch.argmax(_out, 1)
                total += tgt.size(0)
                correct += int((predicts == tgt).sum().detach().cpu())
                # actually it is single class match problem, handle with the simplest method.
                for idx in range(tgt.size(0)):
                    _tgt = int(tgt[idx].detach().cpu().numpy())
                    _pre = int(predicts[idx].detach().cpu().numpy())
                    if _tgt not in class_res: # init
                        class_res[_tgt] = {'fp':0, 'tp':0, 'fn':0}
                    if _pre not in class_res:
                        class_res[_pre] = {'fp':0, 'tp':0, 'fn':0}
                    if _pre != _tgt:
                        class_res[_pre]['fp'] += 1
                        class_res[_tgt]['fn'] += 1
                    else:
                        class_res[_tgt]['tp'] += 1

            
            acc = correct / total

        self.metrics['acc'] = acc
        self.metrics['cls_res'] = class_res
        return self.metrics

class resnetPredictor:
    def __init__(self, device, batch):
        self.device = device
        self.batch = batch

    def _preprocess(self, imgf):
        img = PIL.Image.open(imgf)
        transform=transforms.Compose([transforms.Resize(
        (224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        input = transform(img)
        input = input.unsqueeze(0)
        return input


    def __call__(self, source,  model):
        assert os.path.exists(source)
        img_fs = [os.path.join(source, x) for x in os.listdir(source) if x.endswith('jpg') or x.endswith('png') or x.endswith('jpeg')]
        #yield_batch = batch_preprocess(img_fs, self.batch)
        # 不想写生成器了，batch 暂时当成1吧。
        preds = []
        with torch.no_grad():
            model = model.eval().to(self.device)
            for img in img_fs:
                inp = self._preprocess(img)
                oup = model(inp.to(self.device))
                pred = oup.detach().cpu().numpy()
                preds.append([os.path.basename(img), pred[0]])
        return preds

def get_trainer(**kwargs):
    nc = int(kwargs.pop('nc'))
    suffix = kwargs.pop('suffix')
    data_path = kwargs.pop('data')
    trainer = resnetTrainer(data_path, nc, suffix, **kwargs)
    return trainer

def get_validator(**kwargs):
    data_path = kwargs.get('data')
    batch_size = kwargs.get('batch')
    train_loader, test_loader = load_dataset(data_path, batch_size)
    return resnetValidator(test_loader)

def get_predictor(**kwargs):
    device = kwargs.get('device', 'cpu')
    batch = kwargs.get('batch', 1)
    return resnetPredictor(device, batch)