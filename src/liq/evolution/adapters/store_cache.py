"""Evolution result caching and persistence."""

from __future__ import annotations

from typing import Any


class EvolutionStoreCache:
    """Caches and persists evolution results.

    Provides save/load for evolved programs and evolution state
    to enable warm-starting and result inspection.
    """

    def save(self, data: Any, path: Any) -> None:
        """Save evolution results to persistent storage.

        Args:
            data: Evolution results to save.
            path: Storage path.
        """
        raise NotImplementedError

    def load(self, path: Any) -> Any:
        """Load evolution results from persistent storage.

        Args:
            path: Storage path to load from.

        Returns:
            Loaded evolution results.
        """
        raise NotImplementedError
