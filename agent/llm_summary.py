"""
LLM-powered summary generation node for the API Metrics Report Agent.
"""

import json
from typing import Any

from langchain_openai import ChatOpenAI

from agent.config import OPENAI_API_KEY, OPENAI_MODEL
from agent.state import AgentState

SUMMARY_PROMPT = """You are an API performance analyst. Given the following analysis data,
produce a concise, actionable executive summary in plain English. Highlight the most
critical findings, anomalies, and recommendations.

Traffic Analysis:
{traffic_analysis}

Success & Failure Metrics:
{success_failure_metrics}

Performance Metrics:
{performance_metrics}

Time-Based Insights (anomalies only):
{time_based_anomalies}

Reliability Insights:
{reliability_insights}

Write a clear summary with:
1. Overall system health assessment (1-2 sentences)
2. Key highlights (bullet points)
3. Anomalies and concerns (bullet points)
4. Actionable recommendations (bullet points)

Keep it under 500 words. Be specific with numbers."""


def llm_summary_node(state: AgentState) -> dict[str, Any]:
    """Use LLM to generate a human-readable summary of all KPIs."""
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0.2,
    )

    # Prepare condensed data for the prompt (exclude large daily trends)
    traffic = {**state["traffic_analysis"]}
    traffic.pop("daily_trend", None)

    time_anomalies = state["time_based_insights"].get("anomalies", [])

    prompt = SUMMARY_PROMPT.format(
        traffic_analysis=json.dumps(traffic, indent=2),
        success_failure_metrics=json.dumps(state["success_failure_metrics"], indent=2),
        performance_metrics=json.dumps(state["performance_metrics"], indent=2),
        time_based_anomalies=json.dumps(time_anomalies, indent=2),
        reliability_insights=json.dumps(state["reliability_insights"], indent=2),
    )

    response = llm.invoke(prompt)
    return {"llm_summary": response.content}
