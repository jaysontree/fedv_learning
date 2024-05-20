"""
> yueyijie, jaysonyue@outlook.sg
> 2024/02/02
generate pesudo blind values from shared seed
remark: should generate int as blind value, and change params to fix point number when blinding. 
        problem here is without changing data type of parameters(fp32), adding blind value to params leads to precision digits lost. but converting fp32 to fix point has cost.
        here it is a temporary solution, generate a small random number and mutiply with learning rate to scale down the blinding value
        to close to the gradient update.  
"""

from fate_utils.secure_aggregation_helper import DiffieHellman
from fate_utils.hash import sm3_hash
from randomgen import ChaCha
import numpy as np

from fedv.fl_utils.audit_logger import Auditor

def dh_gen():
    pri = DiffieHellman()
    pub = pri.get_public_key()
    return pri, pub

def asymmetric_encryptor(seeds: dict, local: int, scaler: float, size: int, auditor: Auditor) -> np.float32:
    mask = np.zeros(size, dtype=np.float64)
    for rank, secret in seeds.items():
        assert rank != local 
        seed = int.from_bytes(sm3_hash(secret), byteorder='big')
        secure_rng = np.random.Generator(ChaCha(seed))
        tmp_value = secure_rng.uniform(size=size) * scaler
        obj_encrypted_info = {
            'encrypted': True,
            'random_source': True,
            'source': {'pair': f"{local} {rank}",'secret': secret, 'scaler': scaler},
            'method': "seed = int.from_bytes(sm3_hash(secret), byteorder='big')\n secure_rng = np.random.Generator(ChaCha(seed))\n tmp_value = secure_rng.uniform() * scaler",
            'target': "tmp_value"
        }
        auditor.info(obj_encrypted_info)
        if rank < local:
            mask -= tmp_value
        elif rank > local:
            mask += tmp_value
    auditor.info('computing encryption mask')
    obj_encrypted_info = {
        'enrypted': True,
        'source': {'tmp_value':None, 'local': local, 'rank': None},
        'method': "mask += ((int(local < rank) * 2 - 1) * tmp_value)",
        'target': "mask"
    }
    auditor.info(obj_encrypted_info)
    return np.float32(mask)