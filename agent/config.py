"""
Configuration for the API Metrics Report Agent.

Defines scheduling, database, and model settings.
"""

import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# --- MongoDB Configuration ---
MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING", "")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "api-metrics-db")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "metrics")

# --- OpenAI Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# --- Scheduling Configuration ---
# Mode: "recurrent" or "onetime"
RUN_MODE = os.getenv("RUN_MODE", "recurrent")

# Recurrent schedule: cron expression
# Default: Every Wednesday at 9 PM IST (3:30 PM UTC)
CRON_DAY_OF_WEEK = os.getenv("CRON_DAY_OF_WEEK", "wed")
CRON_HOUR = int(os.getenv("CRON_HOUR", "15"))
CRON_MINUTE = int(os.getenv("CRON_MINUTE", "30"))

# --- Report Output ---
REPORT_OUTPUT_DIR = os.getenv("REPORT_OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "reports"))
