"""
> 2024/02/01
> yueyijie, jaysonyue@outlook.sg
start key exchange server
"""
import grpc
import asyncio
from concurrent import futures
from typing import Iterable
from argparse import ArgumentParser
import threading

import fedv.protobuf.key_exchange_pb2_grpc as key_exchange_pb2_grpc
import fedv.protobuf.key_exchange_pb2 as key_exchange_pb2

# 是否考虑线程安全，是否需要加锁？
stocker = {}
finished = set()
global VERIFICATION_KEY
stop_event = threading.Event()

class KeyExchangerServicer(key_exchange_pb2_grpc.KeyExchangerServicer):
    def PutPubKey(self, requests: Iterable[key_exchange_pb2.PubKey], context):
        for request in requests:
            if request.node_id.job_id != VERIFICATION_KEY:
                return key_exchange_pb2.KeyInfo()
            if request.node_id.pair_alice not in stocker:
                alice = request.node_id.pair_alice
                stocker[request.node_id.pair_alice] = {}
            stocker[request.node_id.pair_alice][request.node_id.pair_bob] = request.pub_key
        print(f'received public key from {alice}')
        return key_exchange_pb2.KeyInfo(status = key_exchange_pb2.KeyInfo.SUCCESS)

    def QueryPubKey(self, request, context):
        if request.pair_alice in stocker:
            if request.pair_bob in stocker[request.pair_alice]:
                return key_exchange_pb2.PubKey(node_id = request, pub_key = stocker[request.pair_alice][request.pair_bob])
        return key_exchange_pb2.PubKey()

    def ExchangeDone(self, request, context):
        if request.job_id == VERIFICATION_KEY:
            finished.add(request.pair_alice)
        if len(finished) == len(stocker.keys()):
            stop_event.set()
            return key_exchange_pb2.ServerInfo(status = key_exchange_pb2.ServerInfo.DONE)
        return key_exchange_pb2.ServerInfo(status = key_exchange_pb2.ServerInfo.WAIT)

def serve(port='37001'):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    key_exchange_pb2_grpc.add_KeyExchangerServicer_to_server(KeyExchangerServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    #server.wait_for_termination()
    stop_event.wait()
    server.stop(2)

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('--addr', required=True)
    parser.add_argument('--job_id', required=True)
    args = parser.parse_args()
    VERIFICATION_KEY = args.job_id
    port = args.addr.split(':')[-1]
    #asyncio.get_event_loop().run_until_complete(serve(port=port))
    serve(port=port)
