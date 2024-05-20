#!/bin/bash

for pid in `ps aux | grep "python -m start_" | grep -v "grep" | awk '{print $2}'`; do
        echo "service process ${pid} found, terminating";
        kill ${pid};
done;

echo "All services stopped"
