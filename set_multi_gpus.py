import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class myDDP(DDP):
    def __getattr__(self, name):
        if name == 'module':
            return self._modules['module']
        else:
            return getattr(self.module, name)
        
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def set_ddp(rank, world_size):
    dist.init_process_group(backend="nccl", 
                            init_method='tcp://127.0.0.1:23456', 
                            rank=rank, 
                            world_size=world_size)
    setup_for_distributed(rank==0)
    torch.cuda.set_device(rank)