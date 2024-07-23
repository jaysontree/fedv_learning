Scripts to simulate Swarm Learning / Federated Learning Experiments.
- remark: These are scripts to run experiments without using Privacy Computation Platform, which is not released to public yet. (Swarm Learning uses Dynamic Aggregator, which requires Privacy Compuation Platform to enable. With/without dynamic aggregator should not affect training results.)
- Example results:
![result](./Results/pred.jpg)

##### Quickstart Guide
Run YOLOv8 Federated Learning:
1. Prepare environment:
please refer to [requirements.txt](../VisualFederated/requirements.txt) or use docker container.
```bash
pip install torch torchvision # match your device
pip install -r requirements
pip install ultralytics==8.0.234 --no-deps # newer version not verified
```
2. [Prepare Dataset](#dataset-preparation):
prepare YOLO dataset and YAML.
3. start [aggregator](#configuration):
```bash
# you may use Tmux/Sceen to run command in sessions. or use nohup to run in background.
# modify configuration/IP&Port accordingly
python server.py
```
4. start key exchanger:
```bash
python exchange_provider.py
```
5. start training [clients](#configuration):
```bash
# server, exchange_provider, clients can run at different place/device, 
# but the ip addr should be correctly indicated.
python client.py
# python client_01.py
# python client_02.py
# python client_03.py
```
then the training should start running.

##### dataset preparation
How to prepare YOLO dataset
- put images in following struct
```bash
├── images
│   ├── test
│   ├── train
│   └── val
└── labels
    ├── test
    ├── train
    └── val
```
- create YAML file
```YAML

path: PATH_TO_YOUR_DATA
train: images/train
val: images/val
test: images/test

names: # LABELS of YOUR DATA
  0: CLASS1
  1: CLASS2
  2: CLASS3
```
For swarm leanring/federated learning, split your data into datasets for every node, eg.
```bash
# Node1
data_01
├── images
│   ├── test
│   ├── train
│   └── val
└── labels
    ├── test
    ├── train
    └── val

# Node2
data_02
├── images
│   ├── test
│   ├── train
│   └── val
└── labels
    ├── test
    ├── train
    └── val
```
```YAML

path: PATH_TO_YOUR_DATA_01
train: images/train
val: images/val
test: images/test

names: # LABELS of YOUR DATA
  0: normal
  1: fracture
```
Then start your trian clients with each sub data set

##### configuration
- change num of clients
- change dataset
- change num of training rounds
- change other hyper parameters
```python
# server.py
if __name__=='__main__':
    # modify "min_fit_clients", "min_available_clients" to change num of clients
    # modify "data_path" to use your own data(data here is used to initiate model. only number of classes and "names" here matters. no acutal image will be loaded)
    # modify "num_rounds" to change number of aggregation round.
    strategy = SecureAggWeighted_Strategy(min_fit_clients=3, min_available_clients=3, initial_parameters=server_init_params(resume=False, data_path="data.yaml"), on_fit_config_fn=fit_config, fit_metrics_aggregation_fn=metric_aggregation)
    server_config = flwr.server.ServerConfig(num_rounds=100)
    flwr.server.start_server(server_address='[::]:40101', config=server_config, strategy=strategy)
```

```python
# client.py
if __name__=='__main__':
    # modify local idx and ranks to change num of clients.
    # length of ranks should equal to num of clients, each client with a unique rank. 
    local = 1
    ranks = [1, 2, 3]
    seeds = dh_exchange('127.0.0.1:37001', local, ranks, 'testing')
    # the third argument, "1e-5" here should be a number close to your model's learning rate.
    # the forth arugument, "100" here should equal to num of training rounds.
    mask = asymmetric_encryptor(seeds, local, 1e-5, 100)
    # max_iter should equal to num of training rounds
    # change "data" as the path of your sub dataset.
    client = flClient(encrypt_mask=mask, max_iter=100, device='cuda:3', data="data_o1.yaml", batch_size=32)
    fl.client.start_numpy_client(server_address='127.0.0.1:40101' ,client=client)
```

##### evaluation
- YOLO model file is an object which wraps checkpoint, ema checkpoint and other metadata. 
- The model file saved from Swarm Learning script is a pytorch checkpoint file. it is equivalent to {YOLO model}.model. Thus you can initiate a YOLO model, and replace with your checkpoint to use YOLO api on valdiation and prediction.
```python
# evaluate centralized model
from ultralytics import YOLO
model = YOLO('PATH_TO_WEIGHTS')
model.val(data='path_to_yaml', max_det=1) # evaluate use val dataset
model.predict(source='path_to_test_set', max_det=1, conf=1e-7, classes=1, save_txt=True, save_conf=True) # predict target class (used for diagnose)

# evaluate swarm learning model
import torch
sd = torch.load('path_to_model_pt')
model.model = sd
# or use model.model.load_state_dict(sd.state_dict())
model.val(data='path_to_yaml', max_det=1)
model.predict(source='path_to_test_set', max_det=1, conf=1e-7, classes=1, save_txt=True, save_conf=True)
```
### Detailed Results and Figures
please refer to the paper.

### Discussion
- when calculate ROC/Sensitivity/Specificity, we only focus on detected fracture results, to simplify the task as a binary classification task. This may not be rigorous. Please let us know if there is a more scientific approach to implement detetcion model and evaluate it for medical purpose.
- No bootstrap conducted due to the time cost. Running a 100-round swarm learning with 1k image per node takes about 4 hours on my device.
- For encryption/blinding the data should be reformat to fixpoint number. However in this study, the modelparams and blind value are are kept in Float32 format, to save conversion time and reduce data transport load(DL models can be large. fixpoint number double/quadruple its size). the blind value is scaled close to gradient changes, so that during add calculation the pricision can be kept. (fp32 only has around 7 pricision digits, eg. 1.213275 x 10e-5. when adding a large number, say 1.000000 x 10e3, the result will be 1.000000 x 10e3, so the useful information is lost, and after secure aggregation the modelparam become 0).
