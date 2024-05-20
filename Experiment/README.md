Scripts to simulate Swarm Learning / Federated Learning Experiments.

##### Quickstart Guide
Run a swarm learning experiment
1. Prepare environment:
please refer to [requirements](../VisualFederated/requirements.txt) or use docker container.
2. Prepare Dataset:
prepare YOLO dataset and YAML.
3. start aggregator:
```bash
# cd ./Simulation
python server.py
```
4. start key exchanger:
```bash
python exchange_provider.py
```
5. start training clients:
```bash
# server, exchange_provider, clients can run at different place/device, but the ip addr should be correctly indicated.
python client.py
# python client_01.py
# python client_02.py
# python client_03.py
```

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
  0: negative
  1: positive
  2: unknown
```
For swarm leanring/federated learning, split your data into datasets, eg.
```bash
data_01
├── images
│   ├── test
│   ├── train
│   └── val
└── labels
    ├── test
    ├── train
    └── val
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
  0: negative
  1: positive
  2: unknown
```
Then configure your trian clients with sub data set

##### configuration
- change num of clients
- change dataset
- change num of training rounds
```python
# server.py
if __name__=='__main__':
    # modify "min_fit_clients", "min_available_clients" to change num of clients
    # modify "data_path" to use your own data(only used 'names' here to initiate model)
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
```python
# evaluate centralized model
from ultralytics import YOLO
model = YOLO('PATH_TO_WEIGHTS')
model.val(data='path_to_yaml', max_det=1) # evaluate use val dataset
model.predict(source='path_to_test_set', max_det=1, conf=1e-7, classes=1) # predict

# evaluate swarm learning model
import torch
sd = torch.load('path_to_model_pt')
model.model = sd
# or use model.model.load_state_dict(sd.state_dict())
model.val(data='path_to_yaml', max_det=1)
model.predict(source='path_to_test_set', max_det=1, conf=1e-7, classes=1)
```