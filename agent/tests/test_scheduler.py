"""
Tests for the scheduler module.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestScheduler:
    @patch("agent.scheduler.run_agent")
    def test_onetime_mode(self, mock_run_agent):
        mock_run_agent.return_value = {
            "report_json": {"report_metadata": {"generated_at": "2025-01-01"}},
            "report_markdown": "",
        }

        from agent.scheduler import _execute_report

        _execute_report()
        mock_run_agent.assert_called_once()

    @patch("agent.scheduler.RUN_MODE", "onetime")
    @patch("agent.scheduler.run_agent")
    def test_run_scheduler_onetime(self, mock_run_agent):
        mock_run_agent.return_value = {
            "report_json": {"report_metadata": {"generated_at": "2025-01-01"}},
            "report_markdown": "",
        }

        from agent.scheduler import run_scheduler

        run_scheduler()
        mock_run_agent.assert_called_once()

    @patch("agent.scheduler.RUN_MODE", "invalid_mode")
    def test_run_scheduler_invalid_mode(self):
        from agent.scheduler import run_scheduler

        with pytest.raises(ValueError, match="Invalid RUN_MODE"):
            run_scheduler()
