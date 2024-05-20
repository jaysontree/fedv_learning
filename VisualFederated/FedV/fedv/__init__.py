import os

__version__ = "1.0"
__basedir__ = os.path.dirname(os.path.abspath(__file__))
__logs_dir__ = os.path.abspath(os.path.join(__basedir__, os.path.pardir, "logs"))
__config_path__ = os.path.abspath(os.path.join(__basedir__, os.path.pardir, "config.properties"))
__data_dir__ = os.path.abspath(os.path.join(__basedir__, os.path.pardir, "data"))

def get_data_dir():
    return __data_dir__

def get_base_dir():
    return __basedir__