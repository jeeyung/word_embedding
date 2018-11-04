import os
import time
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import multiprocessing as mp
from crawl.twitterCrawl import TwitterCrawler
from configuration import get_config
from trainer import train


def crawl(q):
    twitterCrawler = TwitterCrawler()
    twitterCrawler.streamCrawl(["trump"], q)


def init_processes(rank, size, fn, backend='tcp'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


def run(rank, size):
    queue = mp.Queue()
    # for index, number in range():
    crawler = Process(target=crawl, args=(queue,))
    trainer = Process(target=train, args=(get_config(), queue,))
    crawler.start()
    trainer.start()

    # for proc in procs:
    crawler.join()
    trainer.join()


if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()