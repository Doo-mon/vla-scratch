from typing import Iterable, Iterator, List, Sequence

import torch.distributed as dist
from torch.utils.data.sampler import BatchSampler


class DistributedRankAwareBatchSampler(Iterable[Sequence[int]]):
    """Form global batches, then split evenly across ranks."""

    def __init__(
        self,
        sampler: Iterable[int],
        batch_size: int,
        drop_last: bool,
        num_replicas: int | None = None,
        rank: int | None = None,
    ):
        if num_replicas is None:
            if not dist.is_available() or not dist.is_initialized():
                num_replicas = 1
            else:
                num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available() or not dist.is_initialized():
                rank = 0
            else:
                rank = dist.get_rank()

        self.batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        self.num_replicas = num_replicas
        self.process_index = rank
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[Sequence[int]]:
        initial_data: List[int] = []
        batch_to_yield: List[int] = []
        idx = -1

        for idx, batch in enumerate(self.batch_sampler):
            # gather first reps to fill wrap-around buffer
            if not self.drop_last and idx < self.num_replicas:
                initial_data += list(batch)

            if idx % self.num_replicas == self.process_index:
                batch_to_yield = list(batch)

            if idx % self.num_replicas == self.num_replicas - 1 and (
                self.batch_size is None or len(batch) == self.batch_size
            ):
                if batch_to_yield:
                    yield batch_to_yield
                batch_to_yield = []

        # handle tail for non-drop_last
        if not self.drop_last and initial_data:
            if batch_to_yield:
                yield batch_to_yield

    def __len__(self) -> int:
        return len(self.batch_sampler) // self.num_replicas
