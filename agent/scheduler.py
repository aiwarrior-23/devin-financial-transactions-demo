"""
Scheduler for the API Metrics Report Agent.

Supports two modes:
- "onetime": Run the agent once and exit.
- "recurrent": Run the agent on a cron schedule (default: every Wednesday 9 PM IST).
"""

import logging

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from agent.config import CRON_DAY_OF_WEEK, CRON_HOUR, CRON_MINUTE, RUN_MODE
from agent.graph import run_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _execute_report() -> None:
    """Run the agent and log the outcome."""
    logger.info("Starting API metrics report generation...")
    try:
        result = run_agent()
        report_json = result.get("report_json", {})
        metadata = report_json.get("report_metadata", {})
        logger.info("Report generated at %s", metadata.get("generated_at", "unknown"))
        logger.info("Report generation completed successfully.")
    except Exception:
        logger.exception("Report generation failed")


def run_scheduler() -> None:
    """Run in the configured mode (onetime or recurrent)."""
    mode = RUN_MODE.lower().strip()
    logger.info("Agent mode: %s", mode)

    if mode == "onetime":
        _execute_report()
    elif mode == "recurrent":
        scheduler = BlockingScheduler()
        trigger = CronTrigger(
            day_of_week=CRON_DAY_OF_WEEK,
            hour=CRON_HOUR,
            minute=CRON_MINUTE,
        )
        scheduler.add_job(_execute_report, trigger)
        logger.info(
            "Scheduled: every %s at %02d:%02d UTC",
            CRON_DAY_OF_WEEK,
            CRON_HOUR,
            CRON_MINUTE,
        )
        logger.info("Scheduler started. Press Ctrl+C to exit.")
        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduler stopped.")
    else:
        logger.error("Invalid RUN_MODE '%s'. Use 'onetime' or 'recurrent'.", mode)
        raise ValueError(f"Invalid RUN_MODE: {mode}")
