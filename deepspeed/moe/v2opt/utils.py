import torch
from deepspeed import comm as dist

def remove_zero_rows(dispatched_input):
    """
    Removes zero-padded rows along dimension 1 of a tensor (e, c, m)
    and returns the compacted tensor (x, m) along with start indices for each e-group.

    Args:
    dispatched_input (torch.Tensor): Tensor of shape (e, c, m)

    Returns:
    torch.Tensor: Flattened tensor of shape (x, m) with zero-rows removed
    torch.Tensor: Index tensor of shape (e,), where each value is the start index in the flattened tensor
    """
    e, c, m = dispatched_input.shape

    row_sums = dispatched_input.abs().sum(dim=2)  # shape (e, c)
    nonzero_mask = row_sums > 1e-10               # boolean mask, shape (e, c)
    flattened_input = dispatched_input[nonzero_mask]  # shape (x, m)
    input_splits = nonzero_mask.sum(dim=1)  # Shape: (e,)

    return flattened_input, input_splits, nonzero_mask

def restore_zero_rows(flattened_input, nonzero_mask):
    """
    Restores the original (e, c, m) shape by reinserting zero rows 
    into a compacted tensor (x, m).

    Args:
    flattened_input (torch.Tensor): Flattened tensor of shape (x, m)
    input_splits (torch.Tensor): Index tensor of shape (e,), indicating how many rows each group had before padding
    nonzero_mask (torch.Tensor): Boolean mask of shape (e, c), indicating original nonzero row locations

    Returns:
    torch.Tensor: Restored tensor of shape (e, c, m) with zero-padded rows reinserted
    """
    e, c = nonzero_mask.shape
    m = flattened_input.shape[1]  # Get the last dimension size
    
    restored_tensor = torch.zeros((e, c, m), dtype=flattened_input.dtype, device=flattened_input.device)
    restored_tensor[nonzero_mask] = flattened_input
    
    return restored_tensor

def compare_uneven_and_padded(out_uneven, out_padded, dim=2):
    rank = dist.get_rank()
    row_sums = out_padded.abs().sum(dim=2)
    nonzero_mask = row_sums > 1e-9
    filtered_padded = out_padded[nonzero_mask]

    if filtered_padded.shape != out_uneven.shape:
        print(f"[Rank={rank}] Mismatch shapes: uneven={out_uneven.shape}, padded={filtered_padded.shape}")
        return

    try:
        torch.testing.assert_close(out_uneven, filtered_padded, rtol=1e-3, atol=1e-3)
        print(f"[Rank={rank}] Verification PASSED: unbalanced vs. gshard match.")
    except AssertionError as e:
        print(f"[Rank={rank}] Verification FAILED: {e}")
        print(f" **** uneven output on rank {rank} ****")
        print(out_uneven)
        print(f" **** filtered padded output on rank {rank} ****")
        print(filtered_padded)

def compare_tensors(t1, t2):
    rank = dist.get_rank()
    if t1.shape != t2.shape:
        print(f"[Rank={rank}] Mismatch shapes: tensor1={t1.shape}, tensor2={t2.shape}")
        return
    try:
        torch.testing.assert_close(t1, t2, rtol=1e-3, atol=1e-3)
        print(f"[Rank={rank}] Verification PASSED.")
    except AssertionError as e:
        print(f"[Rank={rank}] Verification FAILED: {e}")
        print(f" **** tensor 1 on rank {rank} ****")
        print(t1)
        print(f" **** tensor 2 on rank {rank} ****")
        print(t2)

def print_rank(tensor, name=None, print_func=None, group=None, order=True):
    print_func = (print if print_func is None else print_func)
    if dist.get_rank() == 0:
        print(f"========{name}", flush=True)
    if order:
        for i in range(dist.get_world_size(group)):
            if i == dist.get_rank():
                print(f"rank {dist.get_rank()}: ", end="", flush=True)
                print_func(tensor)
            torch.cuda.synchronize()
            dist.barrier()
    else:
        print(f"rank {dist.get_rank()}: {tensor}", flush=True)
    if dist.get_rank() == 0:
        print(f"========{name} done", flush=True)


def create_group(group_size=None, do_test=False):
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if group_size is None:
        group_size = world_size
    assert world_size % group_size == 0

    _groups = []
    _group = None
    for i in range (world_size // group_size):
        ranks = list(range(group_size * i, group_size * (i + 1)))
        curr_group = dist.new_group(ranks=ranks)
        _groups.append(curr_group)
        dist.barrier()
        if rank in ranks:
            _group = curr_group

    if do_test:
        tensor = torch.full([group_size], fill_value=rank, device='cuda')
        out_tensor = torch.empty_like(tensor)
        dist.all_to_all_single(out_tensor, tensor, group=_group)
        gold = torch.arange(rank // group_size * group_size, rank // group_size * group_size + group_size, device=out_tensor.device)
        assert compare(out_tensor, gold), f"incorrect value on rank {rank}"
        print(f"Distributed test passed with world_size {world_size} and group_size {group_size}")

    return _group


def distributed_setup(do_test=False):
    from mpi4py import MPI
    import socket
    import os

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    hostname = socket.gethostname()

    master_addr = comm.bcast(hostname if rank == 0 else None, root=0)
    master_port = 29500

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    assert torch.cuda.is_available(), "cuda is not available"

    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)

    if rank == 0:
        print(f"MASTER_ADDR = {os.environ['MASTER_ADDR']}")
        print(f"MASTER_PORT = {os.environ['MASTER_PORT']}")
        print(f"WORLD_SIZE  = {os.environ['WORLD_SIZE']}")

    dist.init_distributed(dist_backend="nccl")

    if do_test:
        tensor = torch.full([world_size], fill_value=rank, device='cuda')
        out_tensor = torch.empty_like(tensor)
        dist.all_to_all_single(out_tensor, tensor)
        gold = torch.arange(world_size, device='cuda')
        assert compare(out_tensor, gold), f"incorrect value on rank {rank}"
        print(f"Distributed test passed with world_size {world_size}")