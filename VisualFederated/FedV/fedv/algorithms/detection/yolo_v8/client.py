from collections import OrderedDict
import torch
import flwr as fl
import copy
import time
import json

from .alg_utils import get_trainer
from fedv.security.dh_exchange import dh_exchange
from fedv.security.secure_aggregation import asymmetric_encryptor
from db.task_dao import TaskDao
from utils.consts import ComponentName, TaskStatus, TaskResultType
from fedv.fl_utils.audit_logger import Auditor

class flClient(fl.client.NumPyClient):
    def __init__(self, **kwargs):
        self.max_iter = int(kwargs.get('max_iter'))
        self.encrypt_mask = kwargs.get('encrypt_mask')
        self.batch = int(kwargs.get('batch_size'))
        self.data_path = kwargs.get('data', 'data.yaml')
        self.device = torch.device(kwargs.get('device')) if (kwargs.get('device') and torch.cuda.is_available()) else torch.device('cpu')
        trainer = get_trainer(data=self.data_path, epochs=self.max_iter, batch=self.batch, device=self.device)
        #self.net = trainer.model
        self.trainloader = trainer.train_loader
        self.testloader = trainer.test_loader
        self.num_examples = {'trainset':self.trainloader.dataset.__len__(), 'testset':self.testloader.dataset.__len__()}
        self.optimizer = trainer.optimizer
        self.scheduler = trainer.scheduler
        self.validator = trainer.validator
        self.trainer = trainer
        self.criterion = self.trainer.model.init_criterion()
        self.best_fitness = None
        self.mosaic = True
        self.running_loss = torch.tensor(0.)
        self.web_task_id = kwargs.get('web_task_id')
        self.DAO = TaskDao(self.web_task_id)
        self.DAO.init_task_progress(self.max_iter)
        self.DAO.start_task()
        self.local_round_mark = 0
        self.remote_round_mark = None
        self.web_flow_id = self.DAO.get_flow_id()
        self.auditor = Auditor(self.web_flow_id, self.web_task_id)


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
        print(metrics)
        return metrics

    def get_parameters(self, config):
        params = [v.cpu().numpy()  * self.num_examples['trainset'] + self.encrypt_mask[self.remote_round_mark - 1] for _, v in self.trainer.model.state_dict().items()]
        obj_encrypted_info = {
            'encrypted': True,
            'target_computed': True,
            'source': {'encrypt_mask': self.encrypt_mask[self.remote_round_mark - 1], 'params': "*large state_dict items, not shown"},
            'method': "[v.cpu().numpy()  * self.num_examples['trainset'] + self.encrypt_mask for _, v in self.trainer.model.state_dict().items()]",
            "target": "params"
        }
        self.auditor.info("sending params to aggregator")
        self.auditor.info(obj_encrypted_info)
        # params = [v.cpu().numpy()  * self.num_examples['trainset'] for _, v in self.trainer.model.state_dict().items()]
        return params

    def set_parameters(self, parameters):
        params_dict = zip(self.trainer.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k,v in params_dict})
        self.trainer.model.load_state_dict(state_dict, strict=True)

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
        self.DAO.add_task_progress(1)
        self.local_round_mark += 1
        print(self.local_round_mark, self.remote_round_mark, float(self.running_loss.cpu().detach().numpy()))

        if config.get('current_round') >= self.max_iter - 1:
            # consider as finished
            self.DAO.update_task_status(TaskStatus.SUCCESS)
            self.DAO.finish_task_progress()
            self.DAO.update_serving_model(type=TaskResultType.LOSS, source_component='yolo_v8')
            # the validation process designed on main platform needs to create a new flow,
            # but the result is not tagged between original flow and validation flow.
            # Howerver, it is required to always have a validation report with the model
            # thus another validation process within the same flow is triggered here.
            # these code should be removed if the flow on main platform is designed properly
            from .val import val as lastround_val
            last_round_metric = lastround_val(self.data_path, copy.deepcopy(self.trainer.model), self.device)
            valm = self.DAO.get_task_result(TaskResultType.VAL)
            if valm:
                valm_result = json.loads(valm.result)
            else:
                valm_result = {}
            valm_result.update({'status': "finish", "results": last_round_metric})
            self.DAO.save_task_result(valm_result, ComponentName.DETECTION, type=TaskResultType.VAL)
            # the code above 
        return self.get_parameters(config={}), self.num_examples['trainset'], {'running_loss': float(self.running_loss.cpu().detach().numpy())}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        metrics = self.validate_step()
        fitness = metrics.pop('fitness', -self.running_loss)
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
            self.save_model()
        cur_time_stamp = int(round(time.time() * 1000))
        dm = self.DAO.get_task_result('loss')
        if dm:
            dm_result = json.loads(dm.result)
            data = dm_result.get('data')
        else:
            dm_result, data = {}, {}
        
        data[self.remote_round_mark] =  dict(value=float(self.running_loss.cpu().detach().numpy()), timestamp=cur_time_stamp)
        dm_result.update(data = data)
        self.DAO.save_task_result(task_result=dm_result,component_name=ComponentName.DETECTION, type='loss')

        dacc = self.DAO.get_task_result('acc')
        if dacc:
            dacc_result = json.loads(dacc.result)
            acc_data = dacc_result.get('data')
        else:
            dacc_result, acc_data = {}, {}
        acc_data[self.remote_round_mark] = dict(value=float(metrics.get('metrics/mAP50(B)')), timestamp=cur_time_stamp)
        dacc_result.update(data=acc_data)
        self.DAO.save_task_result(task_result=dacc_result, component_name=ComponentName.DETECTION, type='acc')

        return float(self.running_loss.cpu().detach().numpy()), self.num_examples['testset'], metrics
    
    def save_model(self):
        torch.save(self.trainer.model, 'model.pt')


if __name__=='__main__':
    local = 1
    ranks = [1, 2]
    seeds = dh_exchange('127.0.0.1:37001', local, ranks, 'testing')
    mask = asymmetric_encryptor(seeds, local, 1e-5)
    client = flClient(encrypt_mask=mask, max_iter=100, device='cuda')
    fl.client.start_numpy_client(server_address='127.0.0.1:31101' ,client=client)
