import asyncio
import click
import os
import yaml
import sys

def load_config():
    try:
        with open('deploy_config.yml', 'r') as f:
            default_config = yaml.safe_load(f)
    except Exception as e:
        click.echo(f"load default config failed, {e}")
        raise e
    name = os.environ.get('WORKER_NAME', default_config['worker']['name'])
    submitter_port = int(os.environ.get(
        'WORKER_SUBMITTER_PORT', default_config['worker']['port']))
    host_address = os.environ.get('ENGINE_IP_ADDR', default_config['engine']['ip'])
    port_start = int(os.environ.get('WORKER_PORT_START', default_config['worker']['port_start']))
    port_end = int(os.environ.get('WORKER_PORT_END', default_config['worker']['port_end']))
    max_tasks = os.environ.get('WORKER_MAX_TASKS', default_config['worker']['max_tasks'])
    return name, host_address, submitter_port, max_tasks, port_start, port_end

def start_worker(name, host_address, submitter_port, max_tasks, port_start, port_end, data_base_dir=None):
    """start worker servicer (Internal Services)

    Args:
        name (str):
        worker_ip (str):
        max_tasks (int):
        port_start (int):
        port_end (int):
        data_base_dir (Path):
    """
    from utils import logger
    logger.set_logger(f"worker-{name}")
    from worker import Worker

    loop = asyncio.get_event_loop()
    _worker = Worker(
        host = host_address,
        port = submitter_port,
        max_tasks = max_tasks,
        port_start=port_start,
        port_end=port_end,
        data_dir=data_base_dir
    )

    try:
        loop.run_until_complete(_worker.start())
        click.echo(f"worker_{name} started")
        loop.run_forever()
    except KeyboardInterrupt:
        click.echo("keyboard interrupted")
    finally:
        loop.run_until_complete(_worker.stop())
        loop.run_until_complete(asyncio.sleep(1))
        click.echo(f"worker {name} stopped gracefully")
        loop.close()

if __name__=="__main__":
    sys.path.append('.')
    start_worker(*load_config())