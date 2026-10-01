from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")


def load_required_adapter(loader: Callable[..., T], base, adapter_dir: str, name: str) -> T:
    """Load an evaluation adapter or fail rather than relabeling base-model output."""
    try:
        return loader(base, adapter_dir)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load required adapter for {name}: {adapter_dir}"
        ) from exc
