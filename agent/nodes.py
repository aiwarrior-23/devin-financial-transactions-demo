"""
LangGraph node functions for API metrics analysis.

Each node computes a specific set of KPIs and writes its results
into the shared AgentState.
"""

from collections import defaultdict
from typing import Any

from agent.state import AgentState


def fetch_data_node(state: AgentState) -> dict[str, Any]:
    """Fetch raw metrics from MongoDB."""
    from agent.data_fetcher import fetch_metrics

    raw = fetch_metrics()
    return {"raw_metrics": raw}


def traffic_analysis_node(state: AgentState) -> dict[str, Any]:
    """
    Compute traffic KPIs:
    - Total API calls
    - Daily traffic trend
    - Peak traffic day and value
    - Top 5 busiest APIs (service + endpoint)
    """
    metrics = state["raw_metrics"]

    total_calls = sum(m["request_count"] for m in metrics)

    daily_traffic: dict[str, int] = defaultdict(int)
    api_traffic: dict[str, int] = defaultdict(int)

    for m in metrics:
        daily_traffic[m["date"]] += m["request_count"]
        api_key = f"{m['service']}{m['endpoint']}"
        api_traffic[api_key] += m["request_count"]

    daily_trend = [
        {"date": d, "request_count": c}
        for d, c in sorted(daily_traffic.items())
    ]

    peak_day = max(daily_traffic, key=daily_traffic.get)  # type: ignore[arg-type]
    peak_value = daily_traffic[peak_day]

    top_5_apis = sorted(api_traffic.items(), key=lambda x: x[1], reverse=True)[:5]
    top_5 = [{"api": api, "total_requests": count} for api, count in top_5_apis]

    return {
        "traffic_analysis": {
            "total_api_calls": total_calls,
            "daily_trend": daily_trend,
            "peak_traffic_day": peak_day,
            "peak_traffic_value": peak_value,
            "top_5_busiest_apis": top_5,
        }
    }


def success_failure_node(state: AgentState) -> dict[str, Any]:
    """
    Compute success/failure KPIs:
    - Total success and failure counts
    - Success ratio and failure rate
    - APIs with highest failure rate
    """
    metrics = state["raw_metrics"]

    total_success = sum(m["success_count"] for m in metrics)
    total_failure = sum(m["failure_count"] for m in metrics)
    total = total_success + total_failure

    success_ratio = round((total_success / total) * 100, 2) if total else 0.0
    failure_rate = round((total_failure / total) * 100, 2) if total else 0.0

    api_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"success": 0, "failure": 0, "total": 0}
    )
    for m in metrics:
        api_key = f"{m['service']}{m['endpoint']}"
        api_stats[api_key]["success"] += m["success_count"]
        api_stats[api_key]["failure"] += m["failure_count"]
        api_stats[api_key]["total"] += m["request_count"]

    apis_by_failure_rate = []
    for api, stats in api_stats.items():
        if stats["total"] > 0:
            rate = round((stats["failure"] / stats["total"]) * 100, 2)
            apis_by_failure_rate.append({
                "api": api,
                "failure_count": stats["failure"],
                "total_requests": stats["total"],
                "failure_rate_percent": rate,
            })
    apis_by_failure_rate.sort(key=lambda x: x["failure_rate_percent"], reverse=True)

    return {
        "success_failure_metrics": {
            "total_success_count": total_success,
            "total_failure_count": total_failure,
            "success_ratio_percent": success_ratio,
            "failure_rate_percent": failure_rate,
            "apis_with_highest_failure_rate": apis_by_failure_rate[:5],
        }
    }


def performance_metrics_node(state: AgentState) -> dict[str, Any]:
    """
    Compute performance KPIs:
    - Average memory usage
    - Peak memory usage
    - Correlation between traffic and memory usage
    """
    metrics = state["raw_metrics"]

    memory_values = [m["memory_usage_mb"] for m in metrics]
    avg_memory = round(sum(memory_values) / len(memory_values), 2) if memory_values else 0.0
    peak_memory = max(memory_values) if memory_values else 0

    # Compute Pearson correlation between request_count and memory_usage_mb
    n = len(metrics)
    if n > 1:
        x = [m["request_count"] for m in metrics]
        y = memory_values
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n
        std_x = (sum((xi - mean_x) ** 2 for xi in x) / n) ** 0.5
        std_y = (sum((yi - mean_y) ** 2 for yi in y) / n) ** 0.5
        correlation = round(cov / (std_x * std_y), 4) if std_x * std_y else 0.0
    else:
        correlation = 0.0

    return {
        "performance_metrics": {
            "avg_memory_usage_mb": avg_memory,
            "peak_memory_usage_mb": peak_memory,
            "traffic_memory_correlation": correlation,
        }
    }


def time_based_insights_node(state: AgentState) -> dict[str, Any]:
    """
    Compute time-based insights:
    - Day-wise trend (traffic, success, failure)
    - Identify spikes or anomalies using z-score
    """
    metrics = state["raw_metrics"]

    daily: dict[str, dict[str, int]] = defaultdict(
        lambda: {"traffic": 0, "success": 0, "failure": 0}
    )
    for m in metrics:
        daily[m["date"]]["traffic"] += m["request_count"]
        daily[m["date"]]["success"] += m["success_count"]
        daily[m["date"]]["failure"] += m["failure_count"]

    day_wise_trend = [
        {
            "date": d,
            "traffic": v["traffic"],
            "success": v["success"],
            "failure": v["failure"],
        }
        for d, v in sorted(daily.items())
    ]

    # Detect anomalies using z-score on daily traffic
    traffic_values = [d["traffic"] for d in day_wise_trend]
    anomalies = _detect_anomalies(day_wise_trend, traffic_values, "traffic")

    return {
        "time_based_insights": {
            "day_wise_trend": day_wise_trend,
            "anomalies": anomalies,
        }
    }


def reliability_insights_node(state: AgentState) -> dict[str, Any]:
    """
    Compute reliability insights:
    - Days with unusually high failures
    - APIs with unstable performance (high variance in failure rate)
    """
    metrics = state["raw_metrics"]

    # Daily failure totals
    daily_failures: dict[str, int] = defaultdict(int)
    for m in metrics:
        daily_failures[m["date"]] += m["failure_count"]

    failure_list = [
        {"date": d, "failure_count": c} for d, c in sorted(daily_failures.items())
    ]
    failure_values = [d["failure_count"] for d in failure_list]
    high_failure_days = _detect_anomalies(failure_list, failure_values, "failure_count")

    # API stability: variance in daily failure rate per API
    api_daily_rates: dict[str, list[float]] = defaultdict(list)
    for m in metrics:
        api_key = f"{m['service']}{m['endpoint']}"
        rate = (m["failure_count"] / m["request_count"] * 100) if m["request_count"] else 0.0
        api_daily_rates[api_key].append(rate)

    unstable_apis = []
    for api, rates in api_daily_rates.items():
        if len(rates) > 1:
            mean_rate = sum(rates) / len(rates)
            variance = sum((r - mean_rate) ** 2 for r in rates) / len(rates)
            std_dev = variance ** 0.5
            unstable_apis.append({
                "api": api,
                "mean_failure_rate_percent": round(mean_rate, 2),
                "std_dev_failure_rate": round(std_dev, 2),
            })
    unstable_apis.sort(key=lambda x: x["std_dev_failure_rate"], reverse=True)

    return {
        "reliability_insights": {
            "high_failure_days": high_failure_days,
            "unstable_apis": unstable_apis[:5],
        }
    }


def _detect_anomalies(
    records: list[dict[str, Any]],
    values: list[int | float],
    value_key: str,
    threshold: float = 2.0,
) -> list[dict[str, Any]]:
    """Detect anomalies using z-score method."""
    if len(values) < 2:
        return []
    mean = sum(values) / len(values)
    std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
    if std == 0:
        return []

    anomalies = []
    for record, value in zip(records, values):
        z = (value - mean) / std
        if abs(z) >= threshold:
            anomalies.append({
                "date": record["date"],
                value_key: value,
                "z_score": round(z, 2),
                "direction": "spike" if z > 0 else "dip",
            })
    return anomalies
