"""
Tests for report generation.
"""

import json
import os
import tempfile
from unittest.mock import patch

from agent.report_generator import generate_report_node
from agent.state import AgentState


def _make_full_state() -> AgentState:
    """Build a complete AgentState with all analysis results populated."""
    return AgentState(
        raw_metrics=[],
        traffic_analysis={
            "total_api_calls": 804,
            "daily_trend": [
                {"date": "2025-12-23", "request_count": 302},
                {"date": "2025-12-24", "request_count": 302},
                {"date": "2025-12-25", "request_count": 200},
            ],
            "peak_traffic_day": "2025-12-23",
            "peak_traffic_value": 302,
            "top_5_busiest_apis": [
                {"api": "user-api/login", "total_requests": 200},
                {"api": "payment-api/refund", "total_requests": 164},
            ],
        },
        success_failure_metrics={
            "total_success_count": 774,
            "total_failure_count": 30,
            "success_ratio_percent": 96.27,
            "failure_rate_percent": 3.73,
            "apis_with_highest_failure_rate": [
                {
                    "api": "user-api/login",
                    "failure_count": 20,
                    "total_requests": 200,
                    "failure_rate_percent": 10.0,
                }
            ],
        },
        performance_metrics={
            "avg_memory_usage_mb": 579.2,
            "peak_memory_usage_mb": 800,
            "traffic_memory_correlation": 0.85,
        },
        time_based_insights={
            "day_wise_trend": [
                {"date": "2025-12-23", "traffic": 302, "success": 296, "failure": 6},
                {"date": "2025-12-24", "traffic": 302, "success": 298, "failure": 4},
                {"date": "2025-12-25", "traffic": 200, "success": 180, "failure": 20},
            ],
            "anomalies": [],
        },
        reliability_insights={
            "high_failure_days": [],
            "unstable_apis": [
                {
                    "api": "user-api/login",
                    "mean_failure_rate_percent": 10.0,
                    "std_dev_failure_rate": 0.0,
                }
            ],
        },
        llm_summary="The system is performing well overall with a 96.27% success rate.",
    )


class TestGenerateReport:
    def test_report_json_structure(self):
        state = _make_full_state()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("agent.report_generator.REPORT_OUTPUT_DIR", tmpdir):
                result = generate_report_node(state)

        rj = result["report_json"]
        assert "report_metadata" in rj
        assert "traffic_analysis" in rj
        assert "success_failure_metrics" in rj
        assert "performance_metrics" in rj
        assert "time_based_insights" in rj
        assert "reliability_insights" in rj
        assert "executive_summary" in rj

    def test_report_json_serializable(self):
        state = _make_full_state()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("agent.report_generator.REPORT_OUTPUT_DIR", tmpdir):
                result = generate_report_node(state)

        # Should not raise
        json.dumps(result["report_json"])

    def test_report_markdown_contains_sections(self):
        state = _make_full_state()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("agent.report_generator.REPORT_OUTPUT_DIR", tmpdir):
                result = generate_report_node(state)

        md = result["report_markdown"]
        assert "# API Metrics Analysis Report" in md
        assert "## Executive Summary" in md
        assert "## 1. Traffic Analysis" in md
        assert "## 2. Success & Failure Metrics" in md
        assert "## 3. Performance Metrics" in md
        assert "## 4. Time-Based Insights" in md
        assert "## 5. Reliability Insights" in md

    def test_report_files_created(self):
        state = _make_full_state()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("agent.report_generator.REPORT_OUTPUT_DIR", tmpdir):
                generate_report_node(state)

            files = os.listdir(tmpdir)
            json_files = [f for f in files if f.endswith(".json")]
            md_files = [f for f in files if f.endswith(".md")]
            assert len(json_files) == 1
            assert len(md_files) == 1

    def test_report_markdown_includes_data(self):
        state = _make_full_state()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("agent.report_generator.REPORT_OUTPUT_DIR", tmpdir):
                result = generate_report_node(state)

        md = result["report_markdown"]
        assert "804" in md  # total api calls
        assert "96.27%" in md  # success ratio
        assert "800" in md  # peak memory
        assert "user-api/login" in md  # top api
