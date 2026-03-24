"""
Tests for the analysis node functions.
"""

from agent.nodes import (
    _detect_anomalies,
    performance_metrics_node,
    reliability_insights_node,
    success_failure_node,
    time_based_insights_node,
    traffic_analysis_node,
)
from agent.state import AgentState

SAMPLE_METRICS = [
    {
        "date": "2025-12-23",
        "service": "payment-api",
        "endpoint": "/pay",
        "request_count": 138,
        "success_count": 137,
        "failure_count": 1,
        "avg_response_ms": 174,
        "p95_response_ms": 428,
        "cpu_usage_percent": 66,
        "memory_usage_mb": 498,
    },
    {
        "date": "2025-12-23",
        "service": "payment-api",
        "endpoint": "/refund",
        "request_count": 164,
        "success_count": 159,
        "failure_count": 5,
        "avg_response_ms": 143,
        "p95_response_ms": 311,
        "cpu_usage_percent": 41,
        "memory_usage_mb": 692,
    },
    {
        "date": "2025-12-24",
        "service": "order-api",
        "endpoint": "/create",
        "request_count": 159,
        "success_count": 157,
        "failure_count": 2,
        "avg_response_ms": 127,
        "p95_response_ms": 246,
        "cpu_usage_percent": 79,
        "memory_usage_mb": 398,
    },
    {
        "date": "2025-12-24",
        "service": "order-api",
        "endpoint": "/status",
        "request_count": 143,
        "success_count": 141,
        "failure_count": 2,
        "avg_response_ms": 120,
        "p95_response_ms": 270,
        "cpu_usage_percent": 48,
        "memory_usage_mb": 508,
    },
    {
        "date": "2025-12-25",
        "service": "user-api",
        "endpoint": "/login",
        "request_count": 200,
        "success_count": 180,
        "failure_count": 20,
        "avg_response_ms": 250,
        "p95_response_ms": 500,
        "cpu_usage_percent": 90,
        "memory_usage_mb": 800,
    },
]


def _make_state(metrics: list | None = None) -> AgentState:
    """Build a minimal AgentState with raw_metrics populated."""
    return AgentState(raw_metrics=metrics or SAMPLE_METRICS)


class TestTrafficAnalysis:
    def test_total_api_calls(self):
        result = traffic_analysis_node(_make_state())
        ta = result["traffic_analysis"]
        assert ta["total_api_calls"] == 138 + 164 + 159 + 143 + 200

    def test_peak_traffic_day(self):
        result = traffic_analysis_node(_make_state())
        ta = result["traffic_analysis"]
        # Dec 23: 138+164=302, Dec 24: 159+143=302, Dec 25: 200
        assert ta["peak_traffic_day"] in ("2025-12-23", "2025-12-24")
        assert ta["peak_traffic_value"] == 302

    def test_daily_trend_sorted(self):
        result = traffic_analysis_node(_make_state())
        ta = result["traffic_analysis"]
        dates = [d["date"] for d in ta["daily_trend"]]
        assert dates == sorted(dates)

    def test_top_5_apis(self):
        result = traffic_analysis_node(_make_state())
        ta = result["traffic_analysis"]
        assert len(ta["top_5_busiest_apis"]) <= 5
        # Most requests is user-api/login with 200
        assert ta["top_5_busiest_apis"][0]["api"] == "user-api/login"
        assert ta["top_5_busiest_apis"][0]["total_requests"] == 200


class TestSuccessFailure:
    def test_totals(self):
        result = success_failure_node(_make_state())
        sf = result["success_failure_metrics"]
        assert sf["total_success_count"] == 137 + 159 + 157 + 141 + 180
        assert sf["total_failure_count"] == 1 + 5 + 2 + 2 + 20

    def test_ratios(self):
        result = success_failure_node(_make_state())
        sf = result["success_failure_metrics"]
        total = sf["total_success_count"] + sf["total_failure_count"]
        expected_success_ratio = round((sf["total_success_count"] / total) * 100, 2)
        assert sf["success_ratio_percent"] == expected_success_ratio
        assert sf["failure_rate_percent"] == round(100 - expected_success_ratio, 2)

    def test_highest_failure_rate_api(self):
        result = success_failure_node(_make_state())
        sf = result["success_failure_metrics"]
        top = sf["apis_with_highest_failure_rate"][0]
        # user-api/login has 20/200 = 10% failure rate
        assert top["api"] == "user-api/login"
        assert top["failure_rate_percent"] == 10.0


class TestPerformanceMetrics:
    def test_avg_memory(self):
        result = performance_metrics_node(_make_state())
        pm = result["performance_metrics"]
        expected = round((498 + 692 + 398 + 508 + 800) / 5, 2)
        assert pm["avg_memory_usage_mb"] == expected

    def test_peak_memory(self):
        result = performance_metrics_node(_make_state())
        pm = result["performance_metrics"]
        assert pm["peak_memory_usage_mb"] == 800

    def test_correlation_type(self):
        result = performance_metrics_node(_make_state())
        pm = result["performance_metrics"]
        assert isinstance(pm["traffic_memory_correlation"], float)
        assert -1.0 <= pm["traffic_memory_correlation"] <= 1.0


class TestTimeBasedInsights:
    def test_day_wise_trend(self):
        result = time_based_insights_node(_make_state())
        ti = result["time_based_insights"]
        assert len(ti["day_wise_trend"]) == 3
        dates = [d["date"] for d in ti["day_wise_trend"]]
        assert dates == sorted(dates)

    def test_trend_values(self):
        result = time_based_insights_node(_make_state())
        ti = result["time_based_insights"]
        dec23 = next(d for d in ti["day_wise_trend"] if d["date"] == "2025-12-23")
        assert dec23["traffic"] == 302
        assert dec23["success"] == 296
        assert dec23["failure"] == 6


class TestReliabilityInsights:
    def test_unstable_apis(self):
        result = reliability_insights_node(_make_state())
        ri = result["reliability_insights"]
        assert isinstance(ri["unstable_apis"], list)
        assert len(ri["unstable_apis"]) <= 5

    def test_high_failure_days_type(self):
        result = reliability_insights_node(_make_state())
        ri = result["reliability_insights"]
        assert isinstance(ri["high_failure_days"], list)


class TestDetectAnomalies:
    def test_no_anomalies_uniform(self):
        records = [{"date": f"day{i}", "value": 100} for i in range(10)]
        values = [100] * 10
        assert _detect_anomalies(records, values, "value") == []

    def test_detects_spike(self):
        records = [{"date": f"day{i}", "value": 100} for i in range(10)]
        values = [100] * 10
        # Add a huge spike
        records.append({"date": "day10", "value": 1000})
        values.append(1000)
        anomalies = _detect_anomalies(records, values, "value", threshold=2.0)
        assert len(anomalies) >= 1
        assert anomalies[0]["direction"] == "spike"

    def test_single_record(self):
        records = [{"date": "day0", "value": 100}]
        values = [100]
        assert _detect_anomalies(records, values, "value") == []
