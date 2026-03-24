"""
LangGraph state definition for the API Metrics Report Agent.
"""

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """State that flows through the LangGraph nodes."""

    # Raw data from MongoDB
    raw_metrics: list[dict[str, Any]]

    # Computed KPIs from each analysis node
    traffic_analysis: dict[str, Any]
    success_failure_metrics: dict[str, Any]
    performance_metrics: dict[str, Any]
    time_based_insights: dict[str, Any]
    reliability_insights: dict[str, Any]

    # LLM-generated summary
    llm_summary: str

    # Final outputs
    report_json: dict[str, Any]
    report_markdown: str
