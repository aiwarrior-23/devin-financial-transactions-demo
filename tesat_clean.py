import requests
import time
import logging

# ----------------------------
# CONFIG
# ----------------------------
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('devin_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

API_KEY = "cog_63kyyepd3n5jlhb3gtpersyrp7blvsamw66lg53lly72bqxida2a"
# API_KEY = "cog_nnkveh4ty34ytwrn3tv6nmjbfmy723knj6damkilvjva3wljcvoq"
ORG_ID = "org-6e8ba024cc5b404b96991d3474127d53"

BASE_URL = f"https://api.devin.ai/v3/organizations/{ORG_ID}"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


# ----------------------------
# 1. CREATE PLAYBOOK
# ----------------------------
def create_playbook():
    logger.info("Creating playbook...")
    payload = {
        "title": "Fraud Ring Detection Analyzer 2",
        "body": """
Outcome:
Detect fraud rings in transaction datasets.

Procedure:
1. Load dataset
2. Build interaction graph between accounts
3. Detect clusters of accounts
4. Identify circular flows and cash-out patterns
5. Assign risk levels
6. Generate investigation report
""",
        "macro": "!fraud_ring_analyzer"
    }

    try:
        response = requests.post(
            f"{BASE_URL}/playbooks",
            headers=HEADERS,
            json=payload
        )
        
        logger.info(f"Playbook creation status: {response.status_code}")
        logger.info(f"Playbook creation response: {response.text}")
        
        response.raise_for_status()
        
        playbook_id = response.json()["playbook_id"]
        logger.info(f"Playbook created successfully: {playbook_id}")
        return playbook_id
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error creating playbook: {e}")
        logger.error(f"Response status: {e.response.status_code}")
        logger.error(f"Response text: {e.response.text}")
        logger.error(f"Request URL: {e.response.url}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating playbook: {e}")
        raise


# ----------------------------
# 2. CREATE SESSION
# ----------------------------
def create_session(playbook_id):
    logger.info(f"Creating session with playbook: {playbook_id}")
    payload = {
        "title": "Fraud Ring Detection Task",
        "prompt": """
Analyze the dataset and detect fraud rings.
Generate a fraud investigation report.
""",
        "playbook_id": playbook_id,
        "tags": ["fraud", "paysim"],
        "advanced_mode": "create",
        "create_as_user_id": "google-oauth2|104575584322666593814"
    }

    try:
        response = requests.post(
            f"{BASE_URL}/sessions",
            headers=HEADERS,
            json=payload
        )
        
        logger.info(f"Session creation status: {response.status_code}")
        logger.info(f"Session creation response: {response.text}")
        
        response.raise_for_status()
        
        session_id = response.json()["session_id"]
        logger.info(f"Session created successfully: {session_id}")
        return session_id
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error creating session: {e}")
        logger.error(f"Response status: {e.response.status_code}")
        logger.error(f"Response text: {e.response.text}")
        logger.error(f"Request URL: {e.response.url}")
        logger.error(f"Request payload: {payload}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating session: {e}")
        raise


# ----------------------------
# 3. GET SESSION FROM LIST
# ----------------------------
def get_session_from_list(session_id):
    logger.info(f"Searching for session {session_id} in sessions list")
    try:
        response = requests.get(
            f"{BASE_URL}/sessions",
            headers=HEADERS
        )
        
        logger.info(f"Sessions list status: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"Sessions list failed: {response.text}")
            return None
            
        response.raise_for_status()

        sessions = response.json().get("items", [])
        logger.info(f"Found {len(sessions)} sessions in list")

        for s in sessions:
            if s.get("session_id") == session_id:
                logger.info(f"Found session {session_id} in list")
                return s

        logger.warning(f"Session {session_id} not found in list")
        return None
        
    except Exception as e:
        logger.error(f"Error getting session from list: {e}")
        return None


# ----------------------------
# 4. MONITOR SESSION STATUS
# ----------------------------
def monitor_session_status(session_id, timeout=900):
    logger.info(f"Monitoring session via LIST endpoint: {session_id}")
    start = time.time()

    while True:
        session = get_session_from_list(session_id)

        if not session:
            logger.warning("Session not found in list, retrying...")
            time.sleep(5)
            continue

        status = session.get("status")
        prs = session.get("pull_requests", [])
        acus = session.get("acus_consumed")
        updated_at = session.get("updated_at")

        logger.info(f"Status: {status} | ACUs: {acus} | Updated: {updated_at}")

        if prs:
            pr_url = prs[0].get("pr_url")
            logger.info(f"PR created: {pr_url}")
            return pr_url

        if status in ["finished", "failed", "expired"]:
            logger.info(f"Session ended with status: {status}")
            return None

        if time.time() - start > timeout:
            raise TimeoutError("Timeout waiting for session")

        time.sleep(10)


# ----------------------------
# MAIN ORCHESTRATOR
# ----------------------------
def run():
    logger.info("---- START DEVIN ORCHESTRATION ----")
    
    try:
        # playbook_id = create_playbook()
        playbook_id = "playbook-dd0e0d6d68d544b2852e65b5fa37198e"
        logger.info(f"Using playbook ID: {playbook_id}")

        # session_id = create_session(playbook_id)
        session_id = "b47ba415abda45298c3c78207bad9c7f"
        logger.info(f"Using session ID: {session_id}")

        pr_url = monitor_session_status(session_id)

        # Skip session cleanup due to API permissions
        logger.info("Session cleanup skipped - API key lacks required permissions")
        logger.info("Session will auto-expire or can be managed manually in Devin UI")
        
        logger.info("---- DEVIN ORCHESTRATION COMPLETED ----")
        
    except Exception as e:
        logger.error(f"Orchestration failed: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        raise


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        raise
