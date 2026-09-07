"""DataLoader construction utilities.

A single implementation is shared by training and inference to guarantee
consistent multiprocessing and prefetch semantics.
"""

from __future__ import annotations

from torch_geometric.loader import DataLoader


def build_loader(
    dataset,
    sampler,
    batch_size,
    num_workers,
    pin_memory,
    persistent_workers,
    prefetch_factor,
    mp_context,
    shuffle: bool = False,
):
    """Build a safe `torch_geometric.loader.DataLoader`.

    Important guardrails:
    - If `num_workers == 0`, worker-only kwargs are omitted.
    - Evaluation keeps dataset order; training opts into shuffling explicitly.
    - If a sampler is provided (DDP), shuffling is disabled.
    - `prefetch_factor` is only passed when valid.
    """
    num_workers = int(num_workers)
    shuffle = bool(shuffle) and sampler is None

    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=bool(pin_memory),
        drop_last=False,
    )

    if num_workers > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)

        if prefetch_factor is not None:
            pf = int(prefetch_factor)
            if pf > 0:
                kwargs["prefetch_factor"] = pf

        if mp_context is not None:
            context = str(mp_context).strip()
            if context:
                kwargs["multiprocessing_context"] = context

    return DataLoader(dataset, **kwargs)


__all__ = ["build_loader"]
