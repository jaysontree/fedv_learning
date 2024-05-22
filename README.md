This repo contains a <u>Study of Swarm Learning Application in Medical Reseach</u> and an <u>Up-to-date FL/SL Learning Engine Module</u>
___
#### About Swarm Learning Study
- Computer vision technology are developing fast and AI-assisted medical diagnose are promising. Training a reliable model usually requires a large size of dataset. Collecting medical data can be diffcult and violate privacy.
- Swarm Learning allows multiple parties to collaborate in training model while keeping their data private.

This project proposed a swarm learning solution and verified the effectiveness of swarm-learning model compared with centralized model.


___
#### About Engine Module 

##### Background/motivation of FedV
FedV is aimed to provide swarm learning / federated learning abilities with industry-level/cutting-edge cv models in a more flexiable FL framework.
This project is inspired by  [FedVision](https://github.com/FederatedAI/FedVision) and [Wefe](https://github.com/tianmiantech/WeFe)-VisualFL. The basic idea and a small part of code is inherited from these two projects. Although these two projects provide federated learning modules for cv tasks, they are both based on PaddleFL, which seems to be outdated and nolonger mantained. FedV(this project) uses a simple parameter-server framework([Flower]((https://github.com/adap/flower))), and supports a batch of deep learning algorithms.
##### Architecture
![arch](./VisualFederated/arch.JPG)
FedV provides external services to handle computer vision tasks. It works as a plug in module for the privacy computation platform. It can be deployed independently and bind with one platform. The privacy computation platform will manage participants, collect configurations, generate dataset download url, and coordinates all participants then call API to initiate cv tasks for each participates. 
Master servicer handles request and convert request to local task. then submit to task queue. worker servicer is the consumer, execute process to run the task. When Database enabled, it will verify, synchronize task status/progress to the database. The results/metrics are writen to database.
![arch](./VisualFederated/arch2.JPG)
##### Learning workflow

The core workflow follows the basic parameter-server architecture.
A process is as follows
```mermaid
graph TD
    A(((Member A))) --apply--> A1[initiate resource]
    A1 --> A2[start secure aggreagtor/key exchanger]
    A --submit--> A3[start training client]
    A -..-> B(((Member B)))
    B --submit--> B1[start training client]
    A2 --> A4{{aggregator}}
    A3 --> A5{{train client}}
    B1 --> B2{{train client}}
    A5 <-.sync.-> A4
    B2 <-.sync.-> A4
```


Fedv also provides APIs to converage the whole lifecycle of trained models, such as validation workflow and inference workflow.

At current stage,
- The Privacy Compuating Platform is not open sourced. Please use [Wefe](https://github.com/tianmiantech/WeFe) platform as a reference of privacy compuation platform.
- Features/functions which works with Privacy Computation Platform, such as dynamic aggregator are not included now.



##### Quick start
Containered Build & Deploy
```bash
export DOCKER_BUILDKIT=1
docker build . -t {imagename} -f Deploy/Dockerfile_cpu # CPU Version
docker run -dit --restart=always -v {path to deploy_config.file}:/FedV/deploy_config.yml -v {path to db_config.file}:/FedV/config.properties --network=host --name FedV-Service {imagename}
```
Please refer to [deploy readme](./Deploy/README.md)

##### Run Examples
TODO

##### API
Not released at current stage

### Reference Repo
[Flower](https://github.com/adap/flower)
[Wefe](https://github.com/tianmiantech/WeFe)

