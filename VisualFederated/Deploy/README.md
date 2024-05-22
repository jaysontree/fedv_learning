整个VisualFL模块作为模块打包，方便部署。
启动后容器即为独立的算子引擎，对于平台来说是可拔插的、可以启动多个不同的算子引擎供平台切换。

暂时容器网络使用主机模式，避免做代理或者搞端口映射带来的麻烦。
CPU:
```
docker run -dit --restart=always -v {path to deploy_config.file}:/FedV/deploy_config.yml -v {path to db_config.file}:/FedV/config.properties --network=host --name Visual-Service {imagename}
```

GPU
```
docker run -dit --restart=always --gpus="device=0" -v {path to deploy_config.file}:/FedV/deploy_config.yml -v {path to db_config.file}:/FedV/config.properties --network=host --name Visual-Service {imagename}
```

CPU模式下，由于应用了并行计算加速，默认会根据当前机器逻辑核心数创建并行线程池，会尝试吃满全部算力。建议在启动时绑定cpu核心进行限制，而不是使用时间片方式限制资源。例如一个visualfl engine 绑定8个逻辑核心（4个物理核心）
```
docker run -dit --restart=always --cpuset-cpus=0-7 -v {path to deploy_config.file}:/FedV/deploy_config.yml -v {path to db_config.file}:/FedV/config.properties --network=host --name Visual-Service {imagename}
```