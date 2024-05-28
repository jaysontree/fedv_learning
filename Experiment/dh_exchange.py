import grpc
from typing import List, Dict
import time

import key_exchange_pb2_grpc as key_exchange_pb2_grpc
import key_exchange_pb2  as key_exchange_pb2
from secure_aggregation import dh_gen



def dh_exchange(addr: str, local: int, ranks: List[int], job_id: str) -> Dict:

    channel = grpc.insecure_channel(addr)
    stub = key_exchange_pb2_grpc.KeyExchangerStub(channel)

    private_keychain = {}
    shared_secret = {}
    
    def key_gen():
        for rank in ranks:
            if rank == local:
                continue
            pri, pub = dh_gen()
            private_keychain[rank] = pri
            node_pair = key_exchange_pb2.NodeId(pair_alice=local, pair_bob=rank, job_id=job_id)
            pubkey = key_exchange_pb2.PubKey(node_id = node_pair, pub_key = pub)
            yield pubkey

    
    try:
        res = stub.PutPubKey(key_gen())
        assert res.status == key_exchange_pb2.KeyInfo.SUCCESS
    except:
        raise Exception('secure aggregation initilization failed')

    for rank in ranks:
        if rank == local:
            continue
        node_pair = key_exchange_pb2.NodeId(pair_alice=rank, pair_bob=local)
        num_try = 0
        try:
            while(num_try < 100):
                res = stub.QueryPubKey(node_pair)
                print(len(res.pub_key))
                num_try += 1
                time.sleep(3)
                if len(res.pub_key) >= 32:        
                    break
            pub_key = res.pub_key
            secret = private_keychain[rank].diffie_hellman(pub_key)
            shared_secret[rank] = secret
        except:
            raise Exception('secure aggregation failed/timeout')
    
    try:
        finish_inifo = key_exchange_pb2.NodeId(pair_alice=local, job_id=job_id)
        _ = stub.ExchangeDone(finish_inifo)
    except Exception as e:
        raise Exception(f'Except occured when starting secure aggregagtion: {e}')
    
    return shared_secret
