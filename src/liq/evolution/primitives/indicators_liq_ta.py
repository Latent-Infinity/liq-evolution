"""liq-ta indicator primitives (SMA, EMA, RSI, etc.)."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

import liq_ta
import numpy as np
import polars as pl

from liq.evolution.protocols import IndicatorBackend
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.types import BoolSeries, GPType, ParamSpec, Series

_LIQ_FEATURES_IMPORT_ERROR: Exception | None = None
get_indicator: Any | None = None
get_lq_indicator_metadata: Any | None = None
list_lq_indicators: Any | None = None

try:
    from liq.features.indicators import get_indicator
    from liq.features.indicators import (
        get_indicator_metadata as get_lq_indicator_metadata,
    )
    from liq.features.indicators import list_indicators as list_lq_indicators
except Exception as exc:  # pragma: no cover - exercised by environment guard tests
    _LIQ_FEATURES_IMPORT_ERROR = exc


def _require_liq_features() -> None:
    """Raise a clear error when liq-features is unavailable at runtime."""
    if _LIQ_FEATURES_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "liq.features.indicators is unavailable; install `liq-features` before "
            "using LiqFeaturesBackend or running liq-features-backed registries"
        ) from _LIQ_FEATURES_IMPORT_ERROR


def _list_lq_indicators() -> list[dict[str, Any]]:
    _require_liq_features()
    assert list_lq_indicators is not None
    return list_lq_indicators()


def _get_lq_indicator(name: str) -> Any:
    _require_liq_features()
    assert get_indicator is not None
    return get_indicator(name)


def _get_lq_indicator_metadata(name: str) -> dict[str, Any]:
    _require_liq_features()
    assert get_lq_indicator_metadata is not None
    return get_lq_indicator_metadata(name)


# Candlestick patterns all take (open, high, low, close)
_CDL_PREFIX = "cdl_"


def _legacy_output_suffix(indicator: str, output: str) -> str:
    """Map modern output names to known legacy output suffixes."""
    if indicator == "macd":
        if output in {"macd", "macd_line"}:
            return "macd_line"
        if output in {"signal", "signal_line"}:
            return "signal_line"
        if output in {"histogram", "macdhist"}:
            return "histogram"

    if indicator == "bollinger":
        if output == "upper":
            return "upperband"
        if output == "middle":
            return "middleband"
        if output == "lower":
            return "lowerband"

    if indicator == "aroon":
        if output == "aroon_up":
            return "aroonup"
        if output == "aroon_down":
            return "aroondown"

    return output


def _canonical_output_suffixes(indicator: str, output: str) -> set[str]:
    """Map legacy output names to canonical forms used by `liq.features.indicators`.

    Both legacy and canonical names are returned so callers can register
    aliases for backwards compatibility.
    """
    suffixes = {output}

    if indicator == "bollinger":
        if output in {"upper", "upperband"}:
            suffixes.add("upper")
            suffixes.add("upperband")
        elif output in {"middle", "middleband"}:
            suffixes.add("middle")
            suffixes.add("middleband")
        elif output in {"lower", "lowerband"}:
            suffixes.add("lower")
            suffixes.add("lowerband")

    if indicator == "bollinger" and output in {"upper", "middle", "lower"}:
        # Canonical aliases for legacy Bollinger output names are already
        # accepted in existing tests; keep as-is.
        pass

    if indicator == "macd":
        if output in {"signal", "signal_line", "macdsignal"}:
            suffixes.update({"signal", "signal_line"})
        if output in {"macd", "macd_line"}:
            suffixes.update({"macd", "macd_line"})
        if output in {"histogram", "macdhist"}:
            suffixes.update({"histogram", "macdhist"})

    if indicator in {"stochastic", "stoch"}:
        if output in {"k", "fastk", "slowk", "stoch_k"}:
            suffixes.update({"k", "fastk", "slowk", "stoch_k"})
        elif output in {"d", "fastd", "slowd", "stoch_d"}:
            suffixes.update({"d", "fastd", "slowd", "stoch_d"})

    if output == "real":
        suffixes.add("value")

    if indicator == "aroon":
        if output in {"aroonup", "aroon_up"}:
            suffixes.update({"aroon_up", "aroonup"})
        if output in {"aroondown", "aroon_down"}:
            suffixes.update({"aroon_down", "aroondown"})

    return suffixes


def _normalize_indicator_name(name: str) -> str:
    """Normalize a primitive name and remove redundant prefixes."""
    lower = name.lower()
    if lower.startswith("ta_"):
        lower = lower[3:]
    return lower


def _candidate_name_aliases(name: str) -> list[str]:
    """Return canonical primitive root names."""
    normalized = _normalize_indicator_name(name)
    aliases = {normalized}

    parts = normalized.split("_")
    if len(parts) > 2 and parts[0] == parts[1]:
        collapsed = "_".join([parts[0], *parts[2:]])
        aliases.add(collapsed)

    return sorted(aliases)


def _candidate_output_aliases(indicator: str, output: str) -> list[str]:
    """Return stable output aliases for indicator output registration."""
    aliases = set(_canonical_output_suffixes(indicator, output))
    legacy = _legacy_output_suffix(indicator, output)
    aliases.add(legacy)
    return sorted(aliases)


def _canonical_param_name(param_name: str) -> str:
    """Map legacy indicator parameter names to canonical GP names."""
    canonical = str(param_name).lower()
    aliases = {
        "timeperiod": "period",
        "timeperiod1": "period1",
        "timeperiod2": "period2",
        "timeperiod3": "period3",
        "fastperiod": "fast_period",
        "slowperiod": "slow_period",
        "signalperiod": "signal_period",
        "fastk_period": "k_period",
        "fastd_period": "d_period",
        "slowd_period": "d_period",
    }
    return aliases.get(canonical, canonical)


def _coerce_discrete_default(default: Any, allowed_values: list[Any]) -> Any:
    """Return a concrete default that belongs to `allowed_values`."""
    if not allowed_values:
        return default

    cleaned = allowed_values
    if default in cleaned:
        return default
    if not isinstance(default, (int, float)) or isinstance(default, bool):
        return cleaned[0]

    return min(cleaned, key=lambda value: abs(value - default))


class LiqFeaturesBackend:
    """Indicator backend driven by liq.features metadata and classes."""

    def __init__(self) -> None:
        self._indicators: dict[str, dict[str, Any]] = {
            meta["name"]: meta
            for meta in _list_lq_indicators()
            if isinstance(meta, Mapping) and "name" in meta
        }

        self._cached_indicator_classes: dict[str, type] = {}

    @property
    def indicator_count(self) -> int:
        """Total number of exposed indicators."""
        return len(self._indicators)

    def _get_indicator_meta(self, name: str) -> dict[str, Any]:
        normalized = _normalize_indicator_name(name)
        if normalized not in self._indicators:
            available = ", ".join(sorted(self._indicators)[:10])
            raise ValueError(f"Unknown indicator: {name}. Available: {available}...")
        return self._indicators[normalized]

    @staticmethod
    def _normalize_length(data: dict[str, Any]) -> int:
        if "close" in data:
            return len(data["close"])
        for value in data.values():
            if hasattr(value, "__len__"):
                return len(value)
        raise ValueError("Indicator data input is empty")

    @staticmethod
    def _as_pl_df(data: dict[str, Any]) -> pl.DataFrame:
        frame_data: dict[str, Any] = {}
        n = None
        for key, value in data.items():
            arr = np.asarray(value)
            frame_data[key] = arr
            if n is None:
                n = len(arr)

        if "ts" not in frame_data:
            if n is None:
                raise ValueError("Indicator data input is empty")
            frame_data["ts"] = np.arange(n)

        return pl.DataFrame(frame_data)

    @staticmethod
    def _align_output(
        values: np.ndarray,
        output_ts: np.ndarray | None,
        target_length: int,
    ) -> np.ndarray:
        out = np.full(target_length, np.nan, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        if len(values) == 0:
            return out

        if output_ts is None or len(output_ts) == 0:
            if len(values) >= target_length:
                out[:] = values[:target_length]
            else:
                out[-len(values) :] = values
            return out

        if len(values) == target_length:
            out[:] = values
            return out

        if np.issubdtype(output_ts.dtype, np.integer):
            indices = output_ts.astype(np.int64)
            for idx, value in zip(indices, values, strict=False):
                if 0 <= idx < target_length:
                    out[int(idx)] = float(value)
            return out

        if len(values) < target_length:
            out[-len(values) :] = values
        else:
            out[:] = values[-target_length:]
        return out

    def compute(
        self,
        name: str,
        params: dict[str, Any],
        data: dict[str, Any],
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute an indicator through liq.features classes and align to input length."""
        _ = kwargs.pop("cache", None)
        _ = kwargs.pop("ts", None)
        out = kwargs.pop("out", None)
        output_index = kwargs.pop("output_index", 0)

        if kwargs:
            raise TypeError(f"Unsupported kwargs for LiqFeaturesBackend: {sorted(kwargs)}")

        meta = self._get_indicator_meta(name)
        normalized = _normalize_indicator_name(name)
        cls = self._cached_indicator_classes.get(normalized)
        if cls is None:
            cls = _get_lq_indicator(normalized)
            self._cached_indicator_classes[normalized] = cls

        indicator = cls(params=params)
        df = self._as_pl_df(data)

        result = indicator.compute(df)
        outputs = list(meta.get("outputs", ["value"]))
        if output_index < 0 or output_index >= len(outputs):
            raise IndexError(f"Invalid output index {output_index} for {name}")

        output_name = outputs[output_index]
        if output_name in result.columns:
            result_col = result[output_name]
        elif "value" in result.columns:
            result_col = result["value"]
        elif len(result.columns) == 2 and "ts" in result.columns:
            # Single output with an unexpected alias (backward compatible fallback).
            result_col = result[result.columns[1]]
        else:
            raise KeyError(
                f"Indicator {name} output '{output_name}' is unavailable. "
                f"Available columns: {result.columns}"
            )

        result_values = np.asarray(result_col.to_numpy(), dtype=np.float64)
        result_ts = None
        if "ts" in result.columns:
            result_ts = np.asarray(result["ts"].to_numpy())

        aligned = self._align_output(
            values=result_values,
            output_ts=result_ts,
            target_length=self._normalize_length(data),
        )

        if out is not None:
            out[:] = aligned
            return out
        return aligned

    def list_indicators(
        self,
        category: str | None = None,
    ) -> list[str]:
        if category is None:
            return sorted(self._indicators)

        normalized_category = category.lower()
        return sorted(
            name
            for name, meta in self._indicators.items()
            if normalized_category in str(meta.get("group", "")).lower()
        )


def _build_primitive_name(indicator: str, output: str | None) -> str:
    if output is None:
        return indicator
    return f"{indicator}_{output}"


def _make_param_specs_from_metadata(
    param_metadata: Mapping[str, Any],
) -> list[ParamSpec]:
    """Build ParamSpec list from metadata that may contain defaults."""
    indicator_name = str(param_metadata.get("name", "")).lower()
    if indicator_name.startswith("ta_"):
        indicator_name = indicator_name[3:]

    param_defs = param_metadata.get("parameters", [])
    has_explicit_default_params = isinstance(
        param_metadata.get("default_params"), Mapping
    )
    raw_grid: Mapping[str, Any] = {}

    canonical_param_names: list[str] = []
    raw_param_names: dict[str, str] = {}
    dtype_hints: dict[str, type] = {}
    defaults: dict[str, Any] = {}

    def _register_param_name(raw_name: str) -> str:
        canonical_name = _canonical_param_name(raw_name)
        if not canonical_name or canonical_name in raw_param_names:
            return canonical_name
        canonical_param_names.append(canonical_name)
        raw_param_names[canonical_name] = str(raw_name)
        return canonical_name

    def _coerce_default_value(name: str, value: Any) -> Any:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        if name == "gamma":
            if 0.0 <= float(value) <= 1.0:
                return float(value)
            return None
        return value

    if isinstance(param_defs, Mapping):
        for raw_name, raw_spec in param_defs.items():
            if isinstance(raw_spec, Mapping):
                candidate = raw_spec.get("name", raw_name)
            else:
                candidate = raw_name
            canonical = _canonical_param_name(candidate)
            if not canonical:
                continue
            _register_param_name(candidate)
            if isinstance(raw_spec, Mapping):
                default = _coerce_default_value(canonical, raw_spec.get("default"))
                if default is not None:
                    defaults[canonical] = default
                dtype = raw_spec.get("dtype")
                if str(dtype).lower() in {"int", "integer"}:
                    dtype_hints[canonical] = int
                elif str(dtype).lower() in {"float", "double"}:
                    dtype_hints[canonical] = float
    elif isinstance(param_defs, (list, tuple)):
        for raw_spec in param_defs:
            if isinstance(raw_spec, Mapping):
                if "name" not in raw_spec:
                    continue
                raw_name = str(raw_spec["name"])
                canonical = _register_param_name(raw_name)
                if not canonical:
                    continue
                default = _coerce_default_value(canonical, raw_spec.get("default"))
                if default is not None:
                    defaults[canonical] = default
                dtype = raw_spec.get("dtype")
                if str(dtype).lower() in {"int", "integer"}:
                    dtype_hints[canonical] = int
                elif str(dtype).lower() in {"float", "double"}:
                    dtype_hints[canonical] = float
            elif isinstance(raw_spec, str):
                canonical = _register_param_name(raw_spec)
                _ = canonical

    if isinstance(param_metadata.get("default_params"), Mapping):
        defaults.update(
            {
                canonical_name: value
                for raw_name, value in dict(param_metadata["default_params"]).items()
                if (canonical_name := _canonical_param_name(raw_name))
                and _coerce_default_value(canonical_name, value) is not None
            }
        )

    # If available, use lq indicator defaults for missing entries.
    if indicator_name and not has_explicit_default_params:
        try:
            for raw_name, value in dict(_get_lq_indicator(indicator_name).default_params).items():
                canonical_name = _canonical_param_name(raw_name)
                if canonical_name and canonical_name not in defaults:
                    coerced = _coerce_default_value(canonical_name, value)
                    if coerced is not None:
                        defaults[canonical_name] = coerced
        except Exception:
            pass

    source_params: list[Any] = []
    if not canonical_param_names:
        if isinstance(param_metadata.get("params"), (list, tuple)):
            source_params = list(param_metadata.get("params", []))
        else:
            source_params = list(defaults.keys())

    if source_params:
        for raw_name in source_params:
            canonical = _canonical_param_name(raw_name)
            if not canonical or canonical in raw_param_names:
                continue
            raw_param_names[canonical] = str(raw_name)
            canonical_param_names.append(canonical)

    # Use liq-features parameter grids when available.
    try:
        param_grids = importlib.import_module("liq.features.indicators.param_grids")
        get_param_grid = getattr(param_grids, "get_param_grid", None)
    except Exception:
        get_param_grid = None

    if get_param_grid is not None and indicator_name:
        try:
            raw_grid = get_param_grid(indicator_name)
        except Exception:
            raw_grid = {}

    def _coerce_dtype(name: str, fallback: type | None = None) -> type:
        dtype = dtype_hints.get(name)
        if dtype is int or dtype is float:
            return dtype
        if "column" in name:
            return str
        if (
            name == "gamma"
            or "factor" in name
            or "multiplier" in name
            or "threshold" in name
        ):
            return float
        if (
            name == "lag"
            or name.endswith("_lag")
            or name.endswith("lag")
            or name.endswith("window")
            or "window_" in name
            or name.endswith("length")
            or "length_" in name
            or name.endswith("lookback")
            or "lookback_" in name
            or name == "displacement"
            or name == "offset"
        ):
            return int
        if name.endswith("period") or "period_" in name or name.startswith("period"):
            return int
        if name in {"k", "d"}:
            return int
        if fallback in (int, float):
            return fallback
        return float

    def _infer_default_and_bounds(
        name: str,
        dtype: type,
        provided_default: Any,
    ) -> tuple[int | float, int | float, int | float]:
        if name == "gamma":
            return 0.5, 0.01, 0.99
        if dtype is int:
            if name.endswith("period") or "period" in name:
                return int(provided_default or 14), 1, 500
            return int(provided_default or 14), 1, 500
        if "multiplier" in name or "factor" in name:
            return float(provided_default or 2.0), 0.1, 10.0
        if "threshold" in name:
            return float(provided_default or 50.0), 0.0, 100.0
        return float(provided_default or 1.0), 0.0, 1.0

    def _normalize_grid_values(
        dtype: type,
        values: Any,
    ) -> list[int | float]:
        values_list: list[Any] = list(values or [])
        normalized: list[int | float] = []
        for value in values_list:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            if dtype is int:
                if float(value).is_integer():
                    normalized.append(int(value))
            else:
                normalized.append(float(value))
        return sorted(set(normalized))

    param_specs: list[ParamSpec] = []
    for name in canonical_param_names:
        raw_name = raw_param_names.get(name, name)
        default = defaults.get(name)
        dtype = _coerce_dtype(name, dtype_hints.get(name))
        if dtype is str:
            continue

        allowed_values: list[int | float] | None = None
        if raw_grid is not None:
            for candidate in (name, raw_name):
                values = raw_grid.get(candidate)
                if values is not None and not isinstance(values, Mapping):
                    normalized = _normalize_grid_values(dtype, values)
                    if normalized:
                        allowed_values = normalized
                        break

        if allowed_values is not None:
            chosen_default = _coerce_discrete_default(default, allowed_values)
            try:
                spec = ParamSpec(
                    name,
                    dtype,
                    chosen_default,
                    allowed_values=allowed_values,
                )
            except Exception:
                continue
            param_specs.append(spec)
            continue

        inferred_default, min_value, max_value = _infer_default_and_bounds(
            name,
            dtype,
            default if default is not None else 0,
        )
        if dtype is int:
            inferred_default = int(inferred_default)
            min_value = int(min_value)
            max_value = int(max_value)
        else:
            inferred_default = float(inferred_default)
            min_value = float(min_value)
            max_value = float(max_value)

        if inferred_default < min_value:
            inferred_default = min_value
        if inferred_default > max_value:
            inferred_default = max_value
        try:
            spec = ParamSpec(name, dtype, inferred_default, min_value, max_value)
        except Exception:
            continue
        param_specs.append(spec)

    return param_specs


def _fallback_fallback_indicator_meta() -> dict[str, dict[str, Any]]:
    """Build metadata fallback from the liq-features discovery API."""

    discovered: dict[str, dict[str, Any]] = {}
    try:
        indicators = _list_lq_indicators()
    except Exception:
        return discovered

    for entry in indicators:
        if not isinstance(entry, Mapping):
            continue

        name = str(entry.get("name", "")).lower()
        if not name:
            continue

        inputs = list(entry.get("inputs", ["close"]))
        outputs = list(entry.get("outputs", ["value"]))
        params: list[str] = []

        raw_params = entry.get("parameters", [])
        if isinstance(raw_params, Mapping):
            params = [str(key) for key in raw_params if key]
        elif isinstance(raw_params, list | tuple):
            for param in raw_params:
                if isinstance(param, Mapping) and "name" in param:
                    params.append(str(param["name"]))
                elif isinstance(param, str):
                    params.append(param)

        discovered[name] = {
            "inputs": inputs,
            "outputs": outputs,
            "params": params,
        }

    return discovered


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
    backend: Any,
    name: str,
    input_names: list[str],
    output_index: int,
    *,
    param_names: list[str] | None = None,
    param_dtypes: Mapping[str, type] | None = None,
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
        if len(args) < len(input_names):
            msg = f"{name} expects at least {len(input_names)} inputs, got {len(args)}"
            raise TypeError(msg)

        positional_params = args[len(input_names) :]
        args = args[: len(input_names)]
        data = dict(zip(input_names, args, strict=False))

        if param_names:
            for key, value in zip(param_names, positional_params, strict=False):
                if key not in kwargs:
                    kwargs[key] = value

        if param_dtypes:
            for key, dtype in param_dtypes.items():
                if key not in kwargs:
                    continue
                value = kwargs[key]
                if isinstance(value, bool):
                    continue
                if dtype is int:
                    kwargs[key] = int(round(float(value)))
                elif dtype is float:
                    kwargs[key] = float(value)

        target_len = 0
        for value in data.values():
            if hasattr(value, "__len__"):
                target_len = len(value)
                break

        kwargs, out = _split_kwargs(kwargs)
        try:
            return backend.compute(
                name,
                kwargs,
                data,
                output_index=output_index,
                out=out,
            )
        except ValueError as exc:
            if "insufficient data" in str(exc).lower() and target_len > 0:
                if out is not None:
                    out[:] = np.nan
                    return out
                return np.full(target_len, np.nan, dtype=np.float64)
            raise

    return wrapper


def _backend_indicators(backend: Any) -> dict[str, dict[str, Any]]:
    """Resolve indicator metadata source from a backend wrapper."""

    metadata_source = getattr(backend, "_indicators", None)
    if metadata_source is not None:
        return metadata_source

    nested_backend = getattr(backend, "_backend", None)
    if nested_backend is not None:
        metadata_source = getattr(nested_backend, "_indicators", None)
        if metadata_source is not None:
            return metadata_source

    return _fallback_fallback_indicator_meta()


def _backend_candlestick_patterns(backend: Any) -> list[str]:
    """Resolve candlestick discovery source from a backend wrapper."""
    if isinstance(backend, LiqFeaturesBackend):
        return sorted(
            name
            for name in backend._indicators
            if name.startswith(_CDL_PREFIX)
        )

    nested_backend = getattr(backend, "_backend", None)
    if isinstance(nested_backend, LiqFeaturesBackend):
        return sorted(
            name
            for name in nested_backend._indicators
            if name.startswith(_CDL_PREFIX)
        )

    patterns = getattr(backend, "_candlestick_patterns", None)
    if patterns is not None:
        return patterns

    if nested_backend is not None:
        patterns = getattr(nested_backend, "_candlestick_patterns", None)
        if patterns is not None:
            return patterns

    discovered = []
    for metadata in _list_lq_indicators():
        name = str(metadata.get("name", "")).lower()
        if name.startswith(_CDL_PREFIX):
            discovered.append(name)

    if discovered:
        return sorted(discovered)

    discovered = [
        name
        for name in dir(liq_ta)
        if name.startswith(_CDL_PREFIX) and callable(getattr(liq_ta, name))
    ]
    return sorted(discovered)


class LiqTAIndicatorBackend:
    """Indicator backend powered by liq-ta.

    Auto-discovers indicators and candlestick patterns from the
    liq_ta package.
    """

    def __init__(self) -> None:
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

    def _register_aliases(
        primitive_names: list[str],
        callable_: Any,
        input_types: tuple[GPType, ...],
        *,
        output_type: GPType,
        param_specs: list[ParamSpec] | None,
    ) -> None:
        prototype: Any | None = None
        for primitive_name in primitive_names:
            try:
                existing = registry.get(primitive_name)
            except Exception:
                existing = None
            if existing is not None:
                if prototype is None:
                    prototype = existing
                continue

            if prototype is None:
                registry.register(
                    primitive_name,
                    callable_,
                    category="indicator",
                    input_types=input_types,
                    output_type=output_type,
                    param_specs=param_specs if param_specs else None,
                )
                prototype = registry.get(primitive_name)
            else:
                registry._primitives[primitive_name] = prototype

    # Register standard indicators
    for name, meta in indicators_meta.items():
        root_name = _normalize_indicator_name(name)
        inputs = list(meta["inputs"])
        if root_name == "atr" and inputs == ["close"]:
            inputs = ["high", "low", "close"]
        outputs = meta["outputs"]
        input_types = tuple(Series for _ in inputs)
        try:
            metadata = _get_lq_indicator_metadata(root_name)
            metadata = dict(metadata)
            metadata["name"] = root_name
            if "params" in meta and "parameters" not in metadata:
                metadata["params"] = meta["params"]
            if "outputs" not in metadata:
                metadata["outputs"] = outputs
        except Exception:
            metadata = {"name": root_name, "params": meta["params"], "outputs": outputs}

        param_specs = _make_param_specs_from_metadata(metadata)
        name_aliases = _candidate_name_aliases(root_name)
        output_type = BoolSeries if root_name in candlestick_patterns else Series

        if len(outputs) == 1:
            # Single-output indicator
            param_dtypes = {spec.name: spec.dtype for spec in param_specs}
            wrapper = _make_cached_indicator_callable(
                backend,
                name,
                inputs,
                output_index=0,
                param_names=[spec.name for spec in param_specs],
                param_dtypes=param_dtypes,
            )
            _register_aliases(
                primitive_names=name_aliases,
                callable_=wrapper,
                input_types=input_types,
                output_type=output_type,
                param_specs=param_specs,
            )
        else:
            # Multi-output: register one primitive per output
            param_dtypes = {spec.name: spec.dtype for spec in param_specs}
            for idx, out_name in enumerate(outputs):
                wrapper = _make_cached_indicator_callable(
                    backend,
                    name,
                    inputs,
                    output_index=idx,
                    param_names=[spec.name for spec in param_specs],
                    param_dtypes=param_dtypes,
                )
                output_aliases = _candidate_output_aliases(root_name, out_name)
                output_aliases_normalized = sorted(
                    {_build_primitive_name(alias, out_alias) for alias in name_aliases for out_alias in output_aliases}
                )
                _register_aliases(
                    primitive_names=output_aliases_normalized,
                    callable_=wrapper,
                    input_types=input_types,
                    output_type=output_type,
                    param_specs=param_specs,
                )

    # Register candlestick patterns
    for name in candlestick_patterns:
        wrapper = _make_cached_indicator_callable(
            backend,
            name,
            ["open", "high", "low", "close"],
            output_index=0,
        )
        _register_aliases(
            primitive_names=[name],
            callable_=wrapper,
            input_types=(Series, Series, Series, Series),
            output_type=BoolSeries,
            param_specs=None,
        )
