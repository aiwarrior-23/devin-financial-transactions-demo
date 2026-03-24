"""
MongoDB data fetcher for API metrics.
"""

from typing import Any

from pymongo import MongoClient

from agent.config import MONGO_COLLECTION_NAME, MONGO_CONNECTION_STRING, MONGO_DB_NAME


def get_mongo_client() -> MongoClient:
    """Create and return a MongoDB client."""
    return MongoClient(MONGO_CONNECTION_STRING)


def fetch_metrics(
    client: MongoClient | None = None,
    db_name: str = MONGO_DB_NAME,
    collection_name: str = MONGO_COLLECTION_NAME,
) -> list[dict[str, Any]]:
    """
    Fetch all metric documents from MongoDB.

    Filters out any documents missing the 'service' field (e.g. test documents).
    Returns a list of dicts with keys: date, service, endpoint, request_count,
    success_count, failure_count, avg_response_ms, p95_response_ms,
    cpu_usage_percent, memory_usage_mb.
    """
    should_close = False
    if client is None:
        client = get_mongo_client()
        should_close = True

    try:
        db = client[db_name]
        collection = db[collection_name]
        docs = list(collection.find({"service": {"$exists": True}}))
        for doc in docs:
            doc.pop("_id", None)
        return docs
    finally:
        if should_close:
            client.close()
