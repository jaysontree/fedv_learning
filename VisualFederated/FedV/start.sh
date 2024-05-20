#!/bin/bash
echo "starting FedV services"

sleep 1
nohup python -m start_master 2>&1 &

sleep 1
nohup python -m start_worker 2>&1 &

sleep 1
pid=`ps aux | grep start_master | grep -v "grep" | awk '{print $2}'`
if [[ -z ${pid} ]]; then
        echo "start master failed"
else
        echo "master service pid: ${pid}"
fi

pid=`ps aux | grep start_worker | grep -v "grep" | awk '{print $2}'`
if [[ -z ${pid} ]]; then
        echo "start worker failed";
else
        echo "worker service pid: ${pid}";
fi

sleep infinity