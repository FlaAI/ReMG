"""Background CPU encode prefetch for Ours micro-batches."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Generic, TypeVar


T = TypeVar("T")


class OneAheadPrefetcher(Generic[T]):
    """Keep one prepared micro-batch ahead of the training step.

    processor is not shared across threads.
    """

    def __init__(self, prepare_fn: Callable[[], T]) -> None:
        self._prepare_fn = prepare_fn
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ours-prefetch")
        self._future: Future[T] | None = None
        self._closed = False
        self._submit()

    def _submit(self) -> None:
        if self._closed:
            return
        self._future = self._pool.submit(self._prepare_fn)

    def next(self) -> T:
        """Return the prepared item and kick off the next prepare."""

        if self._future is None:
            raise RuntimeError("Prefetcher has no pending work.")
        item = self._future.result()
        self._submit()
        return item

    def close(self) -> None:
        """Cancel outstanding work and shut down the worker."""

        self._closed = True
        self._pool.shutdown(wait=True, cancel_futures=True)
