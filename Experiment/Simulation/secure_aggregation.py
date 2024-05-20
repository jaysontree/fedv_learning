from fate_utils.secure_aggregation_helper import DiffieHellman
from fate_utils.hash import sm3_hash
from randomgen import ChaCha
import numpy as np


def dh_gen():
    pri = DiffieHellman()
    pub = pri.get_public_key()
    return pri, pub

def asymmetric_encryptor(seeds: dict, local: int, scaler: float, size: int) -> np.float32:
    mask = np.zeros(size, dtype=np.float64)
    for rank, secret in seeds.items():
        assert rank != local 
        seed = int.from_bytes(sm3_hash(secret), byteorder='big')
        secure_rng = np.random.Generator(ChaCha(seed))
        tmp_value = secure_rng.uniform(size=size) * scaler
        if rank < local:
            mask -= tmp_value
        elif rank > local:
            mask += tmp_value
    return np.float32(mask)
