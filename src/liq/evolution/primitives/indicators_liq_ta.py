"""liq-ta indicator primitives (SMA, EMA, RSI, etc.)."""

from __future__ import annotations

from typing import Any

import numpy as np

from liq.evolution.protocols import IndicatorBackend
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.types import BoolSeries, ParamSpec, Series

# Default parameter ranges for GP evolution
_PARAM_RANGES: dict[str, tuple[type, Any, Any, Any]] = {
    # (dtype, default, min, max)
    "period": (int, 14, 2, 200),
    "period1": (int, 7, 2, 100),
    "period2": (int, 14, 2, 100),
    "period3": (int, 28, 2, 100),
    "fast_period": (int, 12, 2, 50),
    "slow_period": (int, 26, 5, 200),
    "signal_period": (int, 9, 2, 50),
    "d_period": (int, 3, 1, 20),
    "k_period": (int, 14, 2, 50),
    "k_slowing": (int, 1, 1, 10),
    "stoch_period": (int, 14, 2, 50),
    "rsi_period": (int, 14, 2, 50),
    "min_period": (int, 2, 2, 30),
    "max_period": (int, 30, 5, 200),
    "fast_limit": (float, 0.5, 0.01, 1.0),
    "slow_limit": (float, 0.05, 0.01, 1.0),
    "vfactor": (float, 0.7, 0.0, 1.0),
    "std_dev": (float, 2.0, 0.5, 4.0),
    "af_start": (float, 0.02, 0.01, 0.1),
    "af_step": (float, 0.02, 0.01, 0.1),
    "af_max": (float, 0.2, 0.1, 0.5),
    "af_init_long": (float, 0.02, 0.01, 0.1),
    "af_init_short": (float, 0.02, 0.01, 0.1),
    "af_long": (float, 0.02, 0.01, 0.1),
    "af_short": (float, 0.02, 0.01, 0.1),
    "af_max_long": (float, 0.2, 0.1, 0.5),
    "af_max_short": (float, 0.2, 0.1, 0.5),
    "offset_on_reverse": (float, 0.0, 0.0, 1.0),
    "start_value": (float, 0.0, 0.0, 1.0),
}

# Candlestick patterns all take (open, high, low, close)
_CDL_PREFIX = "cdl_"


def _make_param_specs(param_names: list[str]) -> list[ParamSpec]:
    """Build ParamSpec list from parameter names using default ranges."""
    specs = []
    for name in param_names:
        if name in _PARAM_RANGES:
            dtype, default, min_val, max_val = _PARAM_RANGES[name]
            specs.append(ParamSpec(name, dtype, default, min_val, max_val))
    return specs


def _coerce_output(result: Any, out: np.ndarray | None = None) -> np.ndarray:
    """Coerce indicator outputs to float64 and optionally reuse an output buffer."""
    if result is None:
        if out is None:
            msg = (
                "Indicator computation returned None and no output buffer was provided."
            )
            raise TypeError(msg)
        return out
    arr = np.asarray(result, dtype=np.float64)
    if out is None:
        return arr
    out[:] = arr
    return out


def _make_single_output_callable(
    func: Any,
    n_inputs: int,
    supports_out: bool,
) -> Any:
    """Create a callable for single-output indicators."""

    def _split_kwargs(
        kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], np.ndarray | None]:
        out = kwargs.pop("out", None)
        return kwargs, out

    if n_inputs == 0:

        def wrapper(**kwargs: Any) -> np.ndarray:
            kwargs, out = _split_kwargs(kwargs)
            if supports_out and out is not None:
                func(out=out, **kwargs)
                return out
            return _coerce_output(func(**kwargs), out)
    elif n_inputs == 1:

        def wrapper(a: np.ndarray, **kwargs: Any) -> np.ndarray:
            kwargs, out = _split_kwargs(kwargs)
            if supports_out and out is not None:
                func(a, out=out, **kwargs)
                return out
            return _coerce_output(func(a, **kwargs), out)
    elif n_inputs == 2:

        def wrapper(a: np.ndarray, b: np.ndarray, **kwargs: Any) -> np.ndarray:
            kwargs, out = _split_kwargs(kwargs)
            if supports_out and out is not None:
                func(a, b, out=out, **kwargs)
                return out
            return _coerce_output(func(a, b, **kwargs), out)
    elif n_inputs == 3:

        def wrapper(
            a: np.ndarray,
            b: np.ndarray,
            c: np.ndarray,
            **kwargs: Any,
        ) -> np.ndarray:
            kwargs, out = _split_kwargs(kwargs)
            if supports_out and out is not None:
                func(a, b, c, out=out, **kwargs)
                return out
            return _coerce_output(func(a, b, c, **kwargs), out)
    else:

        def wrapper(
            a: np.ndarray,
            b: np.ndarray,
            c: np.ndarray,
            d: np.ndarray,
            **kwargs: Any,
        ) -> np.ndarray:
            kwargs, out = _split_kwargs(kwargs)
            if supports_out and out is not None:
                func(a, b, c, d, out=out, **kwargs)
                return out
            return _coerce_output(func(a, b, c, d, **kwargs), out)

    return wrapper


def _make_multi_output_callable(
    func: Any,
    n_inputs: int,
    output_index: int,
    *,
    supports_out: bool = False,
) -> Any:
    """Create a callable that selects one output from a multi-output indicator."""
    _ = supports_out

    def _split_kwargs(
        kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], np.ndarray | None]:
        out = kwargs.pop("out", None)
        return kwargs, out

    if n_inputs == 1:

        def wrapper(a: np.ndarray, **kwargs: Any) -> np.ndarray:
            kwargs, out = _split_kwargs(kwargs)
            result = func(a, **kwargs)
            return _coerce_output(result[output_index], out)
    elif n_inputs == 2:

        def wrapper(a: np.ndarray, b: np.ndarray, **kwargs: Any) -> np.ndarray:
            kwargs, out = _split_kwargs(kwargs)
            result = func(a, b, **kwargs)
            return _coerce_output(result[output_index], out)
    elif n_inputs == 3:

        def wrapper(
            a: np.ndarray,
            b: np.ndarray,
            c: np.ndarray,
            **kwargs: Any,
        ) -> np.ndarray:
            kwargs, out = _split_kwargs(kwargs)
            result = func(a, b, c, **kwargs)
            return _coerce_output(result[output_index], out)
    else:

        def wrapper(
            a: np.ndarray,
            b: np.ndarray,
            c: np.ndarray,
            d: np.ndarray,
            **kwargs: Any,
        ) -> np.ndarray:
            kwargs, out = _split_kwargs(kwargs)
            result = func(a, b, c, d, **kwargs)
            return _coerce_output(result[output_index], out)

    return wrapper


def _make_candlestick_callable(func: Any) -> Any:
    """Create a callable for candlestick pattern indicators (OHLC -> BoolSeries)."""

    def wrapper(
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> np.ndarray:
        result = func(open_, high, low, close)
        return np.asarray(result, dtype=np.float64)

    return wrapper


def _make_cached_indicator_callable(
    backend: IndicatorBackend,
    name: str,
    input_names: list[str],
    output_index: int,
) -> Any:
    """Create a callable that evaluates through an IndicatorBackend.

    This avoids direct ``liq-ta`` function usage and allows caching wrappers
    (e.g. :class:`FeatureContext`) to be used safely.
    """

    def _split_kwargs(
        kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], np.ndarray | None]:
        out = kwargs.pop("out", None)
        return kwargs, out

    def wrapper(*args: np.ndarray, **kwargs: Any) -> np.ndarray:
        if len(args) != len(input_names):
            msg = f"{name} expects {len(input_names)} inputs, got {len(args)}"
            raise TypeError(msg)

        kwargs, out = _split_kwargs(kwargs)
        data = dict(zip(input_names, args, strict=False))
        return backend.compute(name, kwargs, data, output_index=output_index, out=out)

    return wrapper


def _backend_indicators(backend: IndicatorBackend) -> dict[str, dict[str, Any]]:
    """Resolve indicator metadata source from a backend wrapper."""

    metadata_source = getattr(backend, "_indicators", None)
    if metadata_source is not None:
        return metadata_source

    nested_backend = getattr(backend, "_backend", None)
    if nested_backend is not None:
        metadata_source = getattr(nested_backend, "_indicators", None)
        if metadata_source is not None:
            return metadata_source

    import liq_ta

    return dict(liq_ta.INDICATORS)


def _backend_candlestick_patterns(backend: IndicatorBackend) -> list[str]:
    """Resolve candlestick discovery source from a backend wrapper."""

    patterns = getattr(backend, "_candlestick_patterns", None)
    if patterns is not None:
        return patterns

    nested_backend = getattr(backend, "_backend", None)
    if nested_backend is not None:
        patterns = getattr(nested_backend, "_candlestick_patterns", None)
        if patterns is not None:
            return patterns

    import liq_ta

    return [
        name
        for name in sorted(dir(liq_ta))
        if name.startswith(_CDL_PREFIX) and callable(getattr(liq_ta, name))
    ]


class LiqTAIndicatorBackend:
    """Indicator backend powered by liq-ta.

    Auto-discovers indicators and candlestick patterns from the
    liq_ta package.
    """

    def __init__(self) -> None:
        import liq_ta

        self._liq_ta = liq_ta
        self._indicators: dict[str, dict[str, Any]] = dict(liq_ta.INDICATORS)

        # Discover candlestick patterns
        self._candlestick_patterns: list[str] = [
            name
            for name in sorted(dir(liq_ta))
            if name.startswith(_CDL_PREFIX) and callable(getattr(liq_ta, name))
        ]

    @property
    def indicator_count(self) -> int:
        """Total number of discovered indicators (including candlestick patterns)."""
        return len(self._indicators) + len(self._candlestick_patterns)

    @property
    def candlestick_count(self) -> int:
        """Number of discovered candlestick patterns."""
        return len(self._candlestick_patterns)

    def compute(
        self,
        name: str,
        params: dict[str, Any],
        data: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute an indicator value."""
        func = getattr(self._liq_ta, name)
        out = kwargs.pop("out", None)
        output_index = kwargs.pop("output_index", 0)
        if name in self._indicators:
            meta = self._indicators[name]
            inputs = [data[k] for k in meta["inputs"]]
            if meta.get("supports_out", False) and out is not None:
                result = func(*inputs, out=out, **params, **kwargs)
                return _coerce_output(result, out)

            result = func(*inputs, **params, **kwargs)
            if isinstance(result, tuple):
                return _coerce_output(result[output_index], out)
            return _coerce_output(result, out)
        elif name.startswith(_CDL_PREFIX):
            result = func(data["open"], data["high"], data["low"], data["close"])
            return _coerce_output(result, out)
        msg = f"Unknown indicator: {name}"
        raise ValueError(msg)

    def list_indicators(
        self,
        category: str | None = None,
    ) -> list[str]:
        """List available indicator names."""
        names: list[str] = []
        for name, meta in self._indicators.items():
            if category is None or meta.get("category") == category:
                names.append(name)
        if category is None or category == "candlestick":
            names.extend(self._candlestick_patterns)
        return names


def register_liq_ta_indicators(
    registry: PrimitiveRegistry,
    backend: IndicatorBackend,
) -> None:
    """Register liq-ta indicator primitives into the registry.

    Args:
        registry: The GP primitive registry to populate.
        backend: The indicator computation backend.
    """
    indicators_meta = _backend_indicators(backend)
    candlestick_patterns = _backend_candlestick_patterns(backend)

    # Register standard indicators
    for name, meta in indicators_meta.items():
        inputs = meta["inputs"]
        outputs = meta["outputs"]
        param_names = meta["params"]
        input_types = tuple(Series for _ in inputs)
        param_specs = _make_param_specs(param_names)

        if len(outputs) == 1:
            # Single-output indicator
            wrapper = _make_cached_indicator_callable(
                backend, name, inputs, output_index=0
            )
            registry.register(
                f"ta_{name}",
                wrapper,
                category="indicator",
                input_types=input_types,
                output_type=Series,
                param_specs=param_specs if param_specs else None,
            )
        else:
            # Multi-output: register one primitive per output
            for idx, out_name in enumerate(outputs):
                wrapper = _make_cached_indicator_callable(
                    backend,
                    name,
                    inputs,
                    output_index=idx,
                )
                registry.register(
                    f"ta_{name}_{out_name}",
                    wrapper,
                    category="indicator",
                    input_types=input_types,
                    output_type=Series,
                    param_specs=param_specs if param_specs else None,
                )

    # Register candlestick patterns
    for name in candlestick_patterns:
        wrapper = _make_cached_indicator_callable(
            backend,
            name,
            ["open", "high", "low", "close"],
            output_index=0,
        )
        registry.register(
            f"ta_{name}",
            wrapper,
            category="indicator",
            input_types=(Series, Series, Series, Series),
            output_type=BoolSeries,
        )
