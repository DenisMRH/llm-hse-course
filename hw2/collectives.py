"""Кольцевые SUM-операции для одномерных CPU-тензоров и Gloo."""
import torch
import torch.distributed as dist


def ring_reduce_scatter(tensor: torch.Tensor) -> torch.Tensor:
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if tensor.ndim != 1 or tensor.numel() % world_size:
        raise ValueError("Ожидается одномерный тензор длины, кратной world_size")
    chunks = [chunk.clone() for chunk in tensor.chunk(world_size)]
    left, right = (rank - 1) % world_size, (rank + 1) % world_size
    for step in range(world_size - 1):
        # Такой сдвиг оставляет суммарный блок с индексом rank у процесса rank.
        send_index = (rank - step - 1) % world_size
        recv_index = (rank - step - 2) % world_size
        received = torch.empty_like(chunks[recv_index])
        requests = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, chunks[send_index], right),
            dist.P2POp(dist.irecv, received, left),
        ])
        for request in requests:
            request.wait()
        chunks[recv_index].add_(received)
    return chunks[rank]


def ring_all_gather(shard: torch.Tensor) -> torch.Tensor:
    rank, world_size = dist.get_rank(), dist.get_world_size()
    chunks = [torch.empty_like(shard) for _ in range(world_size)]
    chunks[rank].copy_(shard)
    left, right = (rank - 1) % world_size, (rank + 1) % world_size
    for step in range(world_size - 1):
        send_index = (rank - step) % world_size
        recv_index = (rank - step - 1) % world_size
        requests = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, chunks[send_index], right),
            dist.P2POp(dist.irecv, chunks[recv_index], left),
        ])
        for request in requests:
            request.wait()
    return torch.cat(chunks)


def ring_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    return ring_all_gather(ring_reduce_scatter(tensor))
