"""
LangGraph workflow definition for the API Metrics Report Agent.

The graph follows this flow:
  fetch_data -> [traffic_analysis, success_failure, performance_metrics,
                 time_based_insights, reliability_insights] -> llm_summary -> generate_report
"""

from langgraph.graph import END, StateGraph

from agent.llm_summary import llm_summary_node
from agent.nodes import (
    fetch_data_node,
    performance_metrics_node,
    reliability_insights_node,
    success_failure_node,
    time_based_insights_node,
    traffic_analysis_node,
)
from agent.report_generator import generate_report_node
from agent.state import AgentState


def build_graph() -> StateGraph:
    """Build and return the compiled LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("fetch_data", fetch_data_node)
    workflow.add_node("traffic_analysis", traffic_analysis_node)
    workflow.add_node("success_failure", success_failure_node)
    workflow.add_node("performance_metrics", performance_metrics_node)
    workflow.add_node("time_based_insights", time_based_insights_node)
    workflow.add_node("reliability_insights", reliability_insights_node)
    workflow.add_node("llm_summary", llm_summary_node)
    workflow.add_node("generate_report", generate_report_node)

    # Set entry point
    workflow.set_entry_point("fetch_data")

    # After fetching data, fan out to all analysis nodes
    workflow.add_edge("fetch_data", "traffic_analysis")
    workflow.add_edge("fetch_data", "success_failure")
    workflow.add_edge("fetch_data", "performance_metrics")
    workflow.add_edge("fetch_data", "time_based_insights")
    workflow.add_edge("fetch_data", "reliability_insights")

    # All analysis nodes converge to LLM summary
    workflow.add_edge("traffic_analysis", "llm_summary")
    workflow.add_edge("success_failure", "llm_summary")
    workflow.add_edge("performance_metrics", "llm_summary")
    workflow.add_edge("time_based_insights", "llm_summary")
    workflow.add_edge("reliability_insights", "llm_summary")

    # LLM summary feeds into report generation
    workflow.add_edge("llm_summary", "generate_report")

    # Report generation is the final step
    workflow.add_edge("generate_report", END)

    return workflow.compile()


def run_agent() -> dict:
    """Execute the agent graph and return the final state."""
    graph = build_graph()
    result = graph.invoke({})
    return result
