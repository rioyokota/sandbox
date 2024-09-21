import os
import torch
import torch.distributed as dist

def get_rank():
    rank = '0'
    rank = os.getenv('PMI_RANK', rank)
    rank = os.getenv('OMPI_COMM_WORLD_RANK', rank)
    rank = os.getenv('MV2_COMM_WORLD_RANK', rank)
    return int(rank)

def get_world_size():
    size = '1'
    size = os.getenv('PMI_SIZE', size)
    size = os.getenv('OMPI_COMM_WORLD_SIZE', size)
    size = os.getenv('MV2_COMM_WORLD_SIZE', size)
    return int(size)

master_addr = os.getenv('MASTER_ADDR', default='localhost')
master_port = os.getenv('MASTER_PORT', default='8888')
hostname = os.getenv('HOSTNAME', default='localhost')
method = "tcp://{}:{}".format(master_addr, master_port)
rank = get_rank()
world_size = get_world_size()
ngpus = torch.cuda.device_count()
dist.init_process_group("nccl", init_method=method, rank=rank, world_size=world_size)
print('Rank: {}, Size: {}, GPUs: {}, Master: {}, Host: {}'.format(dist.get_rank(), dist.get_world_size(), 
                                                                  ngpus, master_addr, hostname))
device = rank % ngpus
model = torch.nn.Sequential(
    torch.nn.Linear(1,1,bias=False)
).to(device)
model.train()

for param in model.parameters():
    torch.nn.init.constant_(param,1)
    print(param.data)
    dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
    print(param.data)
    param.data /= dist.get_world_size()

dist.destroy_process_group()
