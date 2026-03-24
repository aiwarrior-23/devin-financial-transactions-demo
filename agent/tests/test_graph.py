"""
Tests for the LangGraph workflow construction.
"""

from unittest.mock import MagicMock, patch

from agent.graph import build_graph, run_agent


class TestBuildGraph:
    def test_graph_compiles(self):
        graph = build_graph()
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        graph = build_graph()
        node_names = set(graph.nodes.keys())
        expected = {
            "fetch_data",
            "traffic_analysis",
            "success_failure",
            "performance_metrics",
            "time_based_insights",
            "reliability_insights",
            "llm_summary",
            "generate_report",
            "__start__",
        }
        assert expected.issubset(node_names)


class TestRunAgent:
    @patch("agent.graph.build_graph")
    def test_run_agent_invokes_graph(self, mock_build):
        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = {"report_json": {}, "report_markdown": ""}
        mock_build.return_value = mock_compiled

        result = run_agent()

        mock_compiled.invoke.assert_called_once_with({})
        assert "report_json" in result
        assert "report_markdown" in result
