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
    member_id = os.environ.get('MASTER_NAME', default_config['master']['name'])
    submitter_port = int(os.environ.get(
        'MASTER_SUBMITTER_PORT', default_config['master']['submit_port']))
    worker_address = os.environ.get('ENGINE_IP_ADDR', default_config['engine']['ip']) + ':' + os.environ.get(
        'WORKER_PORT', str(default_config['worker']['port']))
    local = os.environ.get('MASTER_LOCAL_MODE', default_config['master']['local']) in (
        'True', 'true', 'TRUE')
    return member_id, submitter_port, worker_address, local


def start_master(member_id, submitter_port, worker_address, local=False):
    """start master servicer (External Services)

    Args:
        member_id (str):
        submitter_port (int):
        worker_address (str): worker servicer addr (Internal Services)
        local (bool, optional): enable local test mode with fake database connection. Defaults to False.
    """
    from utils import logger
    logger.set_logger(f"master-{member_id}")
    from master import Master

    loop = asyncio.get_event_loop()
    _master = Master(member_id=member_id, worker_address=worker_address,
                     rest_port=submitter_port, local=local)

    try:
        loop.run_until_complete(_master.start())
        click.echo(f"master-{member_id} started")
        loop.run_forever()
    except KeyboardInterrupt:
        click.echo("keyboard interrupted")
    finally:
        loop.run_until_complete(_master.stop())
        loop.run_until_complete(asyncio.sleep(1))
        click.echo(f"master-{member_id} stopped gracefully")
        loop.close()


if __name__ == "__main__":
    sys.path.append('.')
    start_master(*load_config())
