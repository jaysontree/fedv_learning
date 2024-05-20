from collections import OrderedDict
import torch
import flwr as fl
import copy
import time
import json

from alg_utils import get_trainer, EMAmodel
from dh_exchange import dh_exchange
from secure_aggregation import asymmetric_encryptor

class flClient(fl.client.NumPyClient):
    def __init__(self, **kwargs):
        self.max_iter = kwargs.get('max_iter')
        self.encrypt_mask = kwargs.get('encrypt_mask')
        self.device = torch.device(kwargs.get('device')) if (kwargs.get('device') and torch.cuda.is_available()) else torch.device('cpu')
        trainer = get_trainer(data=kwargs.get('data','data.yaml'), epochs=self.max_iter, batch=kwargs.get('batch_size'), device=self.device)
        #self.net = trainer.model
        self.trainloader = trainer.train_loader
        self.testloader = trainer.test_loader
        self.num_examples = {'trainset':self.trainloader.__len__(), 'testset':self.testloader.__len__()}
        self.optimizer = trainer.optimizer
        self.scheduler = trainer.scheduler
        self.validator = trainer.validator
        self.trainer = trainer
        self.criterion = self.trainer.model.init_criterion()
        self.best_fitness = None
        self.mosaic = True
        self.running_loss = torch.tensor(0.)
        self.local_round_mark = 0
        self.remote_round_mark = None
        self.local_ema = None

    def train_one_iter(self):
        for _ in range(1):
            self.running_loss = 0.
            for batch in self.trainloader:
                images = batch['img'].to(self.device).float() / 255
                self.optimizer.zero_grad()
                self.trainer.loss, self.trainer.loss_items = self.criterion(self.trainer.model(images), batch)
                self.running_loss += self.trainer.loss
                self.trainer.loss.backward()
                self.optimizer.step()
            self.running_loss = self.running_loss / len(self.trainloader) / self.trainloader.batch_size
        self.scheduler.step()

    def validate_step(self):
        metrics = self.validator(model=copy.deepcopy(self.trainer.model))
        print('Global Model')
        print(metrics)
        if self.local_ema is not None: 
            metrics1 = self.validator(model=copy.deepcopy(self.local_ema.ema))
            print('EMA')
            print(metrics1)
        return metrics

    def get_parameters(self, config):
        params = [v.cpu().numpy()  * self.num_examples['trainset'] + self.encrypt_mask[self.remote_round_mark - 1] for _, v in self.trainer.model.state_dict().items()]
        # params = [v.cpu().numpy()  * self.num_examples['trainset'] for _, v in self.trainer.model.state_dict().items()]
        return params

    def set_parameters(self, parameters):
        params_dict = zip(self.trainer.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k,v in params_dict})
        #l2diff = torch.tensor(0.)
        #for k, local_weights in  self.trainer.model.state_dict().items():
        #    global_weights = state_dict[k].float()
        #    print(local_weights.size())
        #    print(global_weights.size())
        #    l2diff += torch.square((local_weights.detach().cpu() - global_weights).norm(2))
        #print("||Wg - Wl|| : ", l2diff)
        self.trainer.model.load_state_dict(state_dict, strict=True)
        if self.local_ema is None:
            self.local_ema = EMAmodel(self.trainer.model)
        else:
            self.local_ema.update(self.trainer.model)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        self.remote_round_mark = int(config.get('current_round'))
        if self.mosaic and int(config.get('current_round')) >= self.max_iter - max(10, self.max_iter * 0.3):
            if hasattr(self.trainloader.dataset, 'mosaic'):
                self.trainloader.dataset.mosaic = False
            if hasattr(self.trainloader.dataset, 'close_mosaic'):
                self.trainloader.dataset.close_mosaic(hyp=self.trainer.args)
            self.mosaic = False

        self.train_one_iter()
        self.local_round_mark += 1
        print(self.local_round_mark, self.remote_round_mark, float(self.running_loss.cpu().detach().numpy()))

        if config.get('current_round') > self.max_iter - 1:
            # consider as finished
            metrics = self.validator(model=copy.deepcopy(self.local_ema.ema))
            print(metrics)
            #self.save_ema()

        return self.get_parameters(config={}), self.num_examples['trainset'], {'running_loss': float(self.running_loss.cpu().detach().numpy())}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        metrics = self.validate_step()
        fitness = metrics.pop('fitness', -self.running_loss)
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
            # self.save_model()

        return float(self.running_loss.cpu().detach().numpy()), self.num_examples['testset'], metrics
    
    def save_model(self):
        torch.save(self.trainer.model, 'model.pt')

    def save_ema(self):
        torch.save(self.local_ema.ema, 'model_ema.pt')


if __name__=='__main__':
    local = 3
    ranks = [1, 2, 3]
    seeds = dh_exchange('127.0.0.1:37001', local, ranks, 'testing')
    mask = asymmetric_encryptor(seeds, local, 1e-5, 100)
    client = flClient(encrypt_mask=mask, max_iter=100, device='cuda:2', data="data_o3.yaml", batch_size=32)
    fl.client.start_numpy_client(server_address='127.0.0.1:40101' ,client=client)
