"""Tests for strategy behavior descriptor extraction."""

from __future__ import annotations

import pytest

from liq.evolution.fitness.behavior_descriptors import (
    BEHAVIOR_DESCRIPTOR_BENCHMARK_CORRELATION,
    BEHAVIOR_DESCRIPTOR_DRAWDOWN_PROFILE,
    BEHAVIOR_DESCRIPTOR_HOLDING_PERIOD_PROXY,
    BEHAVIOR_DESCRIPTOR_MAX_LEVERAGE,
    BEHAVIOR_DESCRIPTOR_NET_EXPOSURE,
    BEHAVIOR_DESCRIPTOR_STABILITY,
    BEHAVIOR_DESCRIPTOR_TAIL_RISK,
    BEHAVIOR_DESCRIPTOR_TRADE_FREQUENCY,
    BEHAVIOR_DESCRIPTOR_TURNOVER,
    _benchmark_correlation,
    _clamp,
    _correlation,
    _drawdown_profile,
    _holding_period_proxy,
    _max_drawdown,
    _max_leverage,
    _net_exposure,
    _normalize,
    _safe_returns,
    _stability,
    _tail_risk,
    _to_float_list,
    _trade_frequency,
    extract_behavior_descriptors,
    normalize_behavior_descriptor_value,
)


def _simple_payload() -> dict[str, object]:
    return {
        "position_trace": [1.0, 1.0, 0.0, 2.0],
        "equity_curve": [100.0, 100.0, 99.0, 101.0],
        "pnl_trace": [1.0, 1.5, -0.5, 0.5],
    }


class TestBehaviorDescriptorExtraction:
    def test_extracts_all_descriptors_and_normalizes_values(self) -> None:
        all_descriptors = (
            BEHAVIOR_DESCRIPTOR_HOLDING_PERIOD_PROXY,
            BEHAVIOR_DESCRIPTOR_TURNOVER,
            BEHAVIOR_DESCRIPTOR_NET_EXPOSURE,
            BEHAVIOR_DESCRIPTOR_MAX_LEVERAGE,
            BEHAVIOR_DESCRIPTOR_BENCHMARK_CORRELATION,
            BEHAVIOR_DESCRIPTOR_DRAWDOWN_PROFILE,
            BEHAVIOR_DESCRIPTOR_TRADE_FREQUENCY,
            BEHAVIOR_DESCRIPTOR_TAIL_RISK,
            BEHAVIOR_DESCRIPTOR_STABILITY,
        )

        profile = extract_behavior_descriptors(
            _simple_payload(),
            descriptor_names=all_descriptors,
        )

        assert set(profile.raw.keys()) == set(all_descriptors)
        assert set(profile.normalized.keys()) == set(all_descriptors)
        for key in all_descriptors:
            assert 0.0 <= profile.normalized[key] <= 1.0

    def test_sign_aware_normalizer_maps_full_range(self) -> None:
        assert _normalize(BEHAVIOR_DESCRIPTOR_NET_EXPOSURE, -1.0) == 0.0
        assert _normalize(BEHAVIOR_DESCRIPTOR_NET_EXPOSURE, 0.0) == 0.5
        assert _normalize(BEHAVIOR_DESCRIPTOR_NET_EXPOSURE, 1.0) == 1.0
        assert _normalize(BEHAVIOR_DESCRIPTOR_BENCHMARK_CORRELATION, -1.0) == 0.0
        assert _normalize(BEHAVIOR_DESCRIPTOR_BENCHMARK_CORRELATION, 1.0) == 1.0

    def test_turnover_uses_position_churn_not_capital(self) -> None:
        profile = extract_behavior_descriptors(
            {
                "position_trace": [1.0, 4.0, 2.0],
                "equity_curve": [100.0, 100.0, 100.0],
                "pnl_trace": [1.0, 1.0, -1.0],
            },
            descriptor_names=(BEHAVIOR_DESCRIPTOR_TURNOVER,),
        )

        assert profile.raw[BEHAVIOR_DESCRIPTOR_TURNOVER] == 2.5
        assert profile.normalized[BEHAVIOR_DESCRIPTOR_TURNOVER] == 0.5

    def test_max_drawdown_helper_matches_expected_depth(self) -> None:
        assert _max_drawdown([100.0, 90.0, 95.0, 70.0]) == 0.3

    def test_extracts_requested_subset_and_uses_normalized_values(self) -> None:
        profile = extract_behavior_descriptors(
            _simple_payload(),
            descriptor_names=(
                BEHAVIOR_DESCRIPTOR_TURNOVER,
                BEHAVIOR_DESCRIPTOR_HOLDING_PERIOD_PROXY,
            ),
        )
        assert profile.version == "1.0"
        assert set(profile.raw.keys()) == {
            BEHAVIOR_DESCRIPTOR_TURNOVER,
            BEHAVIOR_DESCRIPTOR_HOLDING_PERIOD_PROXY,
        }
        assert set(profile.normalized.keys()) == {
            BEHAVIOR_DESCRIPTOR_TURNOVER,
            BEHAVIOR_DESCRIPTOR_HOLDING_PERIOD_PROXY,
        }
        assert 0.0 <= profile.normalized[BEHAVIOR_DESCRIPTOR_TURNOVER] <= 1.0
        assert (
            0.0 <= profile.normalized[BEHAVIOR_DESCRIPTOR_HOLDING_PERIOD_PROXY] <= 1.0
        )

    def test_clamps_values_to_unit_interval(self) -> None:
        profile = extract_behavior_descriptors(
            {
                "position_trace": [20.0, 20.0, 20.0],
                "equity_curve": [1.0, 1.0, 1.0],
            },
            descriptor_names=(BEHAVIOR_DESCRIPTOR_MAX_LEVERAGE,),
        )

        assert profile.raw[BEHAVIOR_DESCRIPTOR_MAX_LEVERAGE] == 10.0
        assert profile.normalized[BEHAVIOR_DESCRIPTOR_MAX_LEVERAGE] == 1.0

    def test_nested_trace_payload_is_supported(self) -> None:
        profile = extract_behavior_descriptors(
            {
                "position_trace": {"values": [0.0, 1.0, 1.0, -1.0]},
            },
            descriptor_names=(BEHAVIOR_DESCRIPTOR_HOLDING_PERIOD_PROXY,),
        )
        assert profile.raw[BEHAVIOR_DESCRIPTOR_HOLDING_PERIOD_PROXY] > 0.0

    def test_drawdown_profile_short_equity_is_defaulted(self) -> None:
        profile = extract_behavior_descriptors(
            {"equity_curve": [100.0], "pnl_trace": [1.0], "position_trace": [1.0]},
            descriptor_names=(BEHAVIOR_DESCRIPTOR_DRAWDOWN_PROFILE,),
        )
        assert profile.raw[BEHAVIOR_DESCRIPTOR_DRAWDOWN_PROFILE] == 0.0
        assert profile.normalized[BEHAVIOR_DESCRIPTOR_DRAWDOWN_PROFILE] == 0.0

    def test_unknown_descriptor_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unsupported behavior descriptor"):
            extract_behavior_descriptors(
                _simple_payload(),
                descriptor_names=("not_real",),
            )

    def test_returns_fallback_defaults_when_payload_is_missing(self) -> None:
        profile = extract_behavior_descriptors(
            None,
            descriptor_names=(BEHAVIOR_DESCRIPTOR_TURNOVER,),
        )
        assert profile.raw[BEHAVIOR_DESCRIPTOR_TURNOVER] == 0.0
        assert profile.normalized[BEHAVIOR_DESCRIPTOR_TURNOVER] == 0.0

    def test_descriptor_names_are_reusable_via_exported_constant(self) -> None:
        profile = extract_behavior_descriptors(
            _simple_payload(),
            descriptor_names=(BEHAVIOR_DESCRIPTOR_TURNOVER,),
        )
        assert BEHAVIOR_DESCRIPTOR_TURNOVER in profile.raw

    def test_to_float_list_filters_serialized_payloads(self) -> None:
        payload = _to_float_list({"values": [1, "bad", 3.0, None]})
        assert payload == [1.0, 3.0]


class TestBehaviorDescriptorHelpers:
    """Direct tests for private descriptor helpers to lock behavior."""

    def test_to_float_list_filters_invalid_items_and_infinite_values(self) -> None:
        assert _to_float_list("abc") == []
        assert _to_float_list({"values": [1.0, float("inf"), "bad", None, -2.0]}) == [
            1.0,
            -2.0,
        ]

    def test_to_float_list_empty_and_non_iterable_inputs(self) -> None:
        assert _to_float_list(None) == []
        assert _to_float_list(b"abc") == []

    def test_to_float_list_accepts_nested_mapping_payload(self) -> None:
        assert _to_float_list({"values": [1, "2.0", -3, float("inf")]}) == [
            1.0,
            2.0,
            -3.0,
        ]

    def test_safe_returns_preserves_zero_divider_guard(self) -> None:
        assert _safe_returns([0.0, 2.0, 1.0]) == [0.0, -0.5]

    def test_correlation_returns_zero_with_short_inputs(self) -> None:
        assert _correlation([1.0], [1.0, 2.0]) == 0.0
        assert _correlation([], [1.0, 2.0]) == 0.0

    def test_correlation_returns_zero_when_variance_is_zero(self) -> None:
        assert _correlation([1.0, 1.0, 1.0], [0.0, 1.0, 2.0]) == 0.0

    def test_holding_and_trade_helpers_handle_short_or_flat_positions(self) -> None:
        assert _holding_period_proxy([]) == 0.0
        assert _holding_period_proxy([0.0, 0.0]) == 0.0
        assert _trade_frequency([]) == 0.0
        assert _trade_frequency([1.0]) == 0.0

    def test_net_exposure_and_leverage_calculators_default_to_zero(self) -> None:
        assert _net_exposure([]) == 0.0
        assert _net_exposure([0.0, 0.0]) == 0.0
        assert _max_leverage([1.0, -2.0, 3.0], ()) == 2.0

    def test_clamp_emits_warning_when_adjusting(self) -> None:
        assert _clamp(2.0, 0.0, 1.0) == 1.0

    def test_normalize_behavior_descriptor_value_rejects_unknown_key(self) -> None:
        with pytest.raises(ValueError, match="unsupported behavior descriptor"):
            normalize_behavior_descriptor_value("bad_key", 0.5)

    def test_max_drawdown_caps_to_unit_interval(self) -> None:
        assert _max_drawdown([1.0, -2.0]) == 1.0

    def test_benchmark_correlation_returns_zero_without_benchmark(self) -> None:
        assert _benchmark_correlation([1.0, 2.0], [1.0, 2.0], []) == 0.0

    def test_tail_risk_and_stability_default_to_zero(self) -> None:
        assert _tail_risk([]) == 0.0
        assert _stability([]) == 0.0

    def test_drawdown_profile_short_equity_defaults_to_zero(self) -> None:
        assert _drawdown_profile([10.0]) == 0.0
