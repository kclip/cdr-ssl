import torch
from typing import List, Iterator

def batch_generator(iterable, batch_size: int = 1):
    n = len(iterable)
    for idx in range(0, n, batch_size):
        yield iterable[idx:min(idx + batch_size, n)]


def iterate_on_chunks(list_of_tensors: List[torch.Tensor], n_chunks: int, dim: int = 0) -> Iterator[torch.Tensor]:
    return zip(*[tensor.tensor_split(n_chunks, dim=dim) for tensor in list_of_tensors])
