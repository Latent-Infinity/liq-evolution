"""GP signal provider for liq-signals."""

from __future__ import annotations

from typing import Any


class GPSignalProvider:
    """Generates trading signals from GP-evolved programs.

    Bridges GP program outputs to the liq-signals interface.
    """

    def generate_signals(self, data: Any) -> Any:
        """Generate trading signals from market data.

        Args:
            data: Market data to generate signals from.

        Returns:
            Generated trading signals.
        """
        raise NotImplementedError
