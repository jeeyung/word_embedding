import os
import time
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import multiprocessing as mp
from crawl.twitterCrawl import TwitterCrawler
import argparse
# from configuration import get_config
# from trainer import train
args = argparse.ArgumentParser()
args.add_argument('--rank', default=1, type=int)
args.add_argument('--size', default=3, type=int)

def crawl(rank, size):
    if rank == 1:
        keyword = "trump"

    twitterCrawler = TwitterCrawler()
    twitterCrawler.streamCrawl(["trump"], q)


def init_processes(rank, size, fn, backend='tcp'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = 'deepspark.snu.ac.kr'
    os.environ['MASTER_PORT'] = '22'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

init_processes((args.rank, args.size, crawl))
# def run(rank, size):
#     queue = mp.Queue()
#     # for index, number in range():
#     crawler = Process(target=crawl, args=(queue,))
#     args = get_config()
#     args.queue = queue
#     trainer = Process(target=train, args=(args,))
#     crawler.start()
#     trainer.start()
#
#     # for proc in procs:
#     crawler.join()
#     trainer.join()
#
#
# if __name__ == "__main__":
#     size = 2
#     processes = []
#     # for rank in range(size):
#     p = Process(target=init_processes, args=(rank, size, run))
#     p.start()
#     processes.append(p)
#     for p in processes:
#         p.join()