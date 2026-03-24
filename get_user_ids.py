import requests

API_KEY = "cog_63kyyepd3n5jlhb3gtpersyrp7blvsamw66lg53lly72bqxida2a"
ORG_ID = "org-6e8ba024cc5b404b96991d3474127d53"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Get sessions to see user IDs
response = requests.get(
    f"https://api.devin.ai/v3/organizations/{ORG_ID}/sessions",
    headers=HEADERS
)

if response.status_code == 200:
    sessions = response.json().get("items", [])
    print("Found user IDs from sessions:")
    user_ids = set()
    
    for session in sessions:
        user_id = session.get("user_id")
        if user_id:
            user_ids.add(user_id)
    
    for user_id in sorted(user_ids):
        print(f"  {user_id}")
        
    print(f"\nTotal unique users: {len(user_ids)}")
else:
    print(f"Error: {response.status_code} - {response.text}")
