"""
Report generation nodes: JSON and Markdown outputs.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any

from agent.config import REPORT_OUTPUT_DIR
from agent.state import AgentState


def generate_report_node(state: AgentState) -> dict[str, Any]:
    """Assemble the final JSON and Markdown reports."""
    report_json = _build_json_report(state)
    report_markdown = _build_markdown_report(state, report_json)

    # Write reports to disk
    os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(REPORT_OUTPUT_DIR, f"report_{timestamp}.json")
    md_path = os.path.join(REPORT_OUTPUT_DIR, f"report_{timestamp}.md")

    with open(json_path, "w") as f:
        json.dump(report_json, f, indent=2)

    with open(md_path, "w") as f:
        f.write(report_markdown)

    return {
        "report_json": report_json,
        "report_markdown": report_markdown,
    }


def _build_json_report(state: AgentState) -> dict[str, Any]:
    """Build the structured JSON report."""
    return {
        "report_metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "report_type": "API Metrics Analysis",
        },
        "traffic_analysis": state["traffic_analysis"],
        "success_failure_metrics": state["success_failure_metrics"],
        "performance_metrics": state["performance_metrics"],
        "time_based_insights": state["time_based_insights"],
        "reliability_insights": state["reliability_insights"],
        "executive_summary": state["llm_summary"],
    }


def _build_markdown_report(
    state: AgentState, report_json: dict[str, Any]
) -> str:
    """Build a human-readable Markdown report."""
    traffic = state["traffic_analysis"]
    sf = state["success_failure_metrics"]
    perf = state["performance_metrics"]
    time_ins = state["time_based_insights"]
    rel = state["reliability_insights"]
    summary = state["llm_summary"]

    lines: list[str] = []
    lines.append("# API Metrics Analysis Report")
    lines.append("")
    generated = report_json["report_metadata"]["generated_at"]
    lines.append(f"**Generated:** {generated}")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(summary)
    lines.append("")

    # Traffic Analysis
    lines.append("## 1. Traffic Analysis")
    lines.append("")
    lines.append(f"- **Total API Calls:** {traffic['total_api_calls']:,}")
    lines.append(f"- **Peak Traffic Day:** {traffic['peak_traffic_day']} ({traffic['peak_traffic_value']:,} requests)")
    lines.append("")
    lines.append("### Top 5 Busiest APIs")
    lines.append("")
    lines.append("| API | Total Requests |")
    lines.append("|-----|---------------|")
    for api in traffic["top_5_busiest_apis"]:
        lines.append(f"| {api['api']} | {api['total_requests']:,} |")
    lines.append("")

    # Daily Traffic Trend
    lines.append("### Daily Traffic Trend")
    lines.append("")
    lines.append("| Date | Requests |")
    lines.append("|------|----------|")
    for day in traffic["daily_trend"]:
        lines.append(f"| {day['date']} | {day['request_count']:,} |")
    lines.append("")

    # Success & Failure
    lines.append("## 2. Success & Failure Metrics")
    lines.append("")
    lines.append(f"- **Total Success:** {sf['total_success_count']:,}")
    lines.append(f"- **Total Failure:** {sf['total_failure_count']:,}")
    lines.append(f"- **Success Ratio:** {sf['success_ratio_percent']}%")
    lines.append(f"- **Failure Rate:** {sf['failure_rate_percent']}%")
    lines.append("")
    lines.append("### APIs with Highest Failure Rate")
    lines.append("")
    lines.append("| API | Failures | Total | Failure Rate |")
    lines.append("|-----|----------|-------|-------------|")
    for api in sf["apis_with_highest_failure_rate"]:
        lines.append(
            f"| {api['api']} | {api['failure_count']:,} | "
            f"{api['total_requests']:,} | {api['failure_rate_percent']}% |"
        )
    lines.append("")

    # Performance
    lines.append("## 3. Performance Metrics")
    lines.append("")
    lines.append(f"- **Average Memory Usage:** {perf['avg_memory_usage_mb']} MB")
    lines.append(f"- **Peak Memory Usage:** {perf['peak_memory_usage_mb']} MB")
    lines.append(f"- **Traffic-Memory Correlation:** {perf['traffic_memory_correlation']}")
    lines.append("")

    # Time-Based Insights
    lines.append("## 4. Time-Based Insights")
    lines.append("")
    lines.append("### Day-wise Trend")
    lines.append("")
    lines.append("| Date | Traffic | Success | Failure |")
    lines.append("|------|---------|---------|---------|")
    for day in time_ins["day_wise_trend"]:
        lines.append(
            f"| {day['date']} | {day['traffic']:,} | "
            f"{day['success']:,} | {day['failure']:,} |"
        )
    lines.append("")

    anomalies = time_ins.get("anomalies", [])
    if anomalies:
        lines.append("### Anomalies Detected")
        lines.append("")
        lines.append("| Date | Traffic | Z-Score | Direction |")
        lines.append("|------|---------|---------|-----------|")
        for a in anomalies:
            lines.append(
                f"| {a['date']} | {a['traffic']:,} | {a['z_score']} | {a['direction']} |"
            )
        lines.append("")
    else:
        lines.append("*No significant traffic anomalies detected.*")
        lines.append("")

    # Reliability
    lines.append("## 5. Reliability Insights")
    lines.append("")
    high_fail = rel.get("high_failure_days", [])
    if high_fail:
        lines.append("### Days with Unusually High Failures")
        lines.append("")
        lines.append("| Date | Failures | Z-Score | Direction |")
        lines.append("|------|----------|---------|-----------|")
        for d in high_fail:
            lines.append(
                f"| {d['date']} | {d['failure_count']:,} | "
                f"{d['z_score']} | {d['direction']} |"
            )
        lines.append("")
    else:
        lines.append("*No days with unusually high failure counts detected.*")
        lines.append("")

    unstable = rel.get("unstable_apis", [])
    if unstable:
        lines.append("### APIs with Unstable Performance")
        lines.append("")
        lines.append("| API | Mean Failure Rate | Std Dev |")
        lines.append("|-----|------------------|---------|")
        for api in unstable:
            lines.append(
                f"| {api['api']} | {api['mean_failure_rate_percent']}% | "
                f"{api['std_dev_failure_rate']} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("*Report generated by API Metrics LangGraph Agent*")
    lines.append("")

    return "\n".join(lines)
