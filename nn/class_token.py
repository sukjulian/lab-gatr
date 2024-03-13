from typing import Callable
import torch
from xformers.ops.fmha import BlockDiagonalMask
from torch_scatter import segment_csr


def class_token_forward_wrapper(forward: Callable) -> Callable:

    def class_token_forward(x: torch.Tensor, **kwargs) -> torch.Tensor:
        attention_mask = kwargs.pop('attention_mask') if 'attention_mask' in kwargs else None

        if isinstance(attention_mask, BlockDiagonalMask):
            ptr = attention_mask.q_seqinfo.seqstart

            x = make_sandwich(x, ptr)
            attention_mask = BlockDiagonalMask.from_seqlens((ptr[1:] - ptr[:-1] + 1).tolist())

            kwargs = {key: make_sandwich(value, ptr) for key, value in kwargs.items()}

        elif isinstance(attention_mask, torch.Tensor):
            x = torch.cat((x.mean(dim=0, keepdim=True), x))

            attention_mask = attention_mask.to_sparse().coalesce()

            coo = attention_mask.indices()
            coo[-2:, :] += 1

            attention_mask = torch.sparse_coo_tensor(coo, attention_mask.values())

            kwargs = {key: torch.cat((value.mean(dim=0, keepdim=True), value)) for key, value in kwargs.items()}

        else:
            x = torch.cat((x.mean(dim=0, keepdim=True), x))
            kwargs = {key: torch.cat((value.mean(dim=0, keepdim=True), value)) for key, value in kwargs.items()}

        x = forward(x, attention_mask=attention_mask, **kwargs)

        # GATr support
        if isinstance(x, tuple):
            class_token = (tensor[ptr[:-1]] if isinstance(attention_mask, BlockDiagonalMask) else tensor[0:1] for tensor in x)

        else:
            class_token = x[ptr[:-1]] if isinstance(attention_mask, BlockDiagonalMask) else x[0:1]

        return class_token

    return class_token_forward


def make_sandwich(x: torch.Tensor, ptr: torch.Tensor) -> torch.Tensor:
    t = segment_csr(x, ptr.long().to(x.device), reduce='mean')

    sandwich = []
    for i in range(ptr.numel() - 1):
        sandwich.extend((t[slice(i, i + 1)], x[slice(*ptr[slice(i, i + 2)])]))

    return torch.cat(sandwich)
