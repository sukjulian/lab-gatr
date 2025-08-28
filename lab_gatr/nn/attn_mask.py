import torch
from xformers.ops.fmha import BlockDiagonalMask


def get_attn_mask(target_batch: torch.Tensor, source_batch=None):

    if target_batch is None:
        attn_mask = None

    else:
        attn_mask = BlockDiagonalMask.from_seqlens(
            q_seqlen=torch.bincount(target_batch).tolist(),
            kv_seqlen=torch.bincount(source_batch).tolist() if source_batch is not None else None
        ).to(target_batch.device)

    return attn_mask
