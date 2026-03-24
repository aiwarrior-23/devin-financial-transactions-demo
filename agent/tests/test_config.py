"""
Tests for configuration module.
"""

from agent.config import (
    CRON_DAY_OF_WEEK,
    CRON_HOUR,
    CRON_MINUTE,
    MONGO_COLLECTION_NAME,
    MONGO_DB_NAME,
    OPENAI_MODEL,
    REPORT_OUTPUT_DIR,
    RUN_MODE,
)


class TestConfig:
    def test_default_mongo_db_name(self):
        assert MONGO_DB_NAME == "api-metrics-db"

    def test_default_collection_name(self):
        assert MONGO_COLLECTION_NAME == "metrics"

    def test_default_model(self):
        assert OPENAI_MODEL == "gpt-4o-mini"

    def test_default_run_mode(self):
        assert RUN_MODE in ("onetime", "recurrent")

    def test_default_cron_schedule(self):
        # Default: Wednesday 15:30 UTC = 9:00 PM IST
        assert CRON_DAY_OF_WEEK == "wed"
        assert CRON_HOUR == 15
        assert CRON_MINUTE == 30

    def test_report_output_dir_set(self):
        assert REPORT_OUTPUT_DIR is not None
        assert len(REPORT_OUTPUT_DIR) > 0
