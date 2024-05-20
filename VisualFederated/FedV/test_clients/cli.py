import urllib.parse
from pathlib import Path

import click
import aiohttp
import asyncio
import json
import yaml

@click.group()
def cli():
    ...


def post(endpoint, path, json_data):
    async def post_co():
        url = urllib.parse.urljoin(f"http://{endpoint}", path)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=json_data
            ) as resp:
                print(resp.status)
                print(json.dumps(await resp.json(), indent=2))
                resp.raise_for_status()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(post_co())



@cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--endpoint",
    type=str,
    required=True,
)
def apply(endpoint, config):
    base = Path(config)
    with base.open("r") as f:
        config_json = json.load(f)
    job_id = config_json.get("job_id")
    task_id = config_json.get("task_id")
    job_type = config_json.get("job_type")
    role = config_json.get("role")
    member_id = config_json.get("member_id")
    callback_url = config_json.get("callback_url")
    env = config_json.get("env")
    data_set = config_json.get("data_set")
    algorithm_config = config_json.get("algorithm_config")

    post(
        endpoint,
        "apply",
        dict(
            job_id=job_id,
            task_id=task_id,
            job_type=job_type,
            role=role,
            member_id=member_id,
            env=env,
            data_set=data_set,
            algorithm_config=algorithm_config,
            callback_url=callback_url
        )
    )

@cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--endpoint",
    type=str,
    required=True,
)
def submit(endpoint, config):
    base  = Path(config)
    with base.open("r") as f:
        config_json = json.load(f)
    job_id = config_json.get("job_id")
    task_id = config_json.get("task_id")
    job_type = config_json.get("job_type")
    role = config_json.get("role")
    member_id = config_json.get("member_id")
    env = config_json.get("env")
    data_set = config_json.get("data_set")
    algorithm_config = config_json.get("algorithm_config")

    post(
        endpoint,
        "submit",
        dict(
            job_id=job_id,
            task_id=task_id,
            job_type=job_type,
            role=role,
            member_id=member_id,
            env=env,
            data_set=data_set,
            algorithm_config=algorithm_config,
        )
    )    

@cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--endpoint",
    type=str,
    required=True,
)
def val(endpoint, config):
    base  = Path(config)
    with base.open("r") as f:
        config_json = json.load(f)
    job_id = config_json.get("job_id")
    task_id = config_json.get("task_id")
    job_type = config_json.get("job_type")
    oot_id = config_json.get("oot_job_id")
    role = config_json.get("role")
    member_id = config_json.get("member_id")
    env = config_json.get("env")
    data_set = config_json.get("data_set")
    algorithm_config = config_json.get("algorithm_config")

    post(
        endpoint,
        "val",
        dict(
            job_id=job_id,
            task_id=task_id,
            job_type=job_type,
            oot_job_id=oot_id,
            role=role,
            member_id=member_id,
            env=env,
            data_set=data_set,
            algorithm_config=algorithm_config,
        )
    )    

@cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--endpoint",
    type=str,
    required=True,
)
def stop(endpoint, config):
    base  = Path(config)
    with base.open("r") as f:
        config_json = json.load(f)
    job_id = config_json.get("job_id")
    post(
        endpoint,
        "stop",
        dict(
            job_id=job_id
        )
    )    

if __name__ == "__main__":
    cli()
