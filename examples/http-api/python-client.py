"""
Keyoku HTTP API - Python Client Example

A simple client demonstrating core keyoku-server operations using the requests library.

Prerequisites:
    pip install requests

Usage:
    export KEYOKU_SESSION_TOKEN="my-secret-token"
    python python-client.py
"""

import os
import sys

import requests

BASE_URL = os.getenv("KEYOKU_BASE_URL", "http://localhost:18900/api/v1")
TOKEN = os.getenv("KEYOKU_SESSION_TOKEN", "")

if not TOKEN:
    print("Error: KEYOKU_SESSION_TOKEN environment variable is required.")
    print("Set it to the same value used when starting keyoku-server.")
    sys.exit(1)

# Common headers for all authenticated requests.
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}

ENTITY_ID = "user-python-example"
AGENT_ID = "python-client"


def health_check():
    """Check server health (no auth required)."""
    print("=== Health Check ===")
    resp = requests.get(f"{BASE_URL}/health")
    resp.raise_for_status()
    data = resp.json()
    print(f"  Status: {data['status']}")
    print(f"  Timestamp: {data['timestamp']}")
    print()


def remember(content: str, source: str = "python-example") -> dict:
    """Extract and store memories from content."""
    print(f"=== Remember ===")
    print(f"  Content: {content[:80]}...")
    resp = requests.post(
        f"{BASE_URL}/remember",
        headers=HEADERS,
        json={
            "entity_id": ENTITY_ID,
            "content": content,
            "agent_id": AGENT_ID,
            "source": source,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    print(f"  Created: {data['memories_created']}")
    print(f"  Updated: {data['memories_updated']}")
    print(f"  Skipped: {data['skipped']}")
    print()
    return data


def search(query: str, limit: int = 5, mode: str = "balanced") -> list:
    """Semantic search across stored memories."""
    print(f"=== Search: '{query}' ===")
    resp = requests.post(
        f"{BASE_URL}/search",
        headers=HEADERS,
        json={
            "entity_id": ENTITY_ID,
            "query": query,
            "limit": limit,
            "mode": mode,
            "min_score": 0.1,
        },
    )
    resp.raise_for_status()
    results = resp.json()
    for i, r in enumerate(results, 1):
        mem = r["memory"]
        print(f"  {i}. [score={r['score']:.2f}] {mem['content']} (type={mem.get('type', 'unknown')})")
    if not results:
        print("  No results found.")
    print()
    return results


def list_memories(limit: int = 10) -> list:
    """List all memories for the entity."""
    print("=== List Memories ===")
    resp = requests.get(
        f"{BASE_URL}/memories",
        headers=HEADERS,
        params={"entity_id": ENTITY_ID, "limit": limit},
    )
    resp.raise_for_status()
    memories = resp.json()
    print(f"  Total: {len(memories)} memories")
    for i, m in enumerate(memories, 1):
        print(f"  {i}. [{m.get('type', '?')}] {m['content']}")
    print()
    return memories


def delete_memory(memory_id: str):
    """Delete a single memory by ID."""
    print(f"=== Delete Memory: {memory_id} ===")
    resp = requests.delete(
        f"{BASE_URL}/memories/{memory_id}",
        headers=HEADERS,
    )
    resp.raise_for_status()
    print(f"  Deleted successfully.")
    print()


def heartbeat_check() -> dict:
    """Run a zero-token heartbeat check for actionable signals."""
    print("=== Heartbeat Check ===")
    resp = requests.post(
        f"{BASE_URL}/heartbeat/check",
        headers=HEADERS,
        json={
            "entity_id": ENTITY_ID,
            "deadline_window": "24h",
            "importance_floor": 0.5,
            "max_results": 10,
            "agent_id": AGENT_ID,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    print(f"  Should act: {data.get('should_act', False)}")
    print(f"  Summary: {data.get('summary', 'none')}")
    if data.get("priority_action"):
        print(f"  Priority action: {data['priority_action']}")
    print()
    return data


def create_schedule(content: str, cron_tag: str) -> dict:
    """Create a recurring scheduled memory."""
    print(f"=== Create Schedule ===")
    print(f"  Content: {content}")
    print(f"  Cron tag: {cron_tag}")
    resp = requests.post(
        f"{BASE_URL}/schedule",
        headers=HEADERS,
        json={
            "entity_id": ENTITY_ID,
            "agent_id": AGENT_ID,
            "content": content,
            "cron_tag": cron_tag,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    print(f"  Created schedule: {data.get('id', 'unknown')}")
    print()
    return data


def list_schedules() -> list:
    """List all active schedules."""
    print("=== List Schedules ===")
    resp = requests.get(
        f"{BASE_URL}/scheduled",
        headers=HEADERS,
        params={"entity_id": ENTITY_ID, "agent_id": AGENT_ID},
    )
    resp.raise_for_status()
    schedules = resp.json()
    if schedules:
        for i, s in enumerate(schedules, 1):
            print(f"  {i}. {s['content']} (tags={s.get('tags', [])})")
    else:
        print("  No active schedules.")
    print()
    return schedules


def get_stats() -> dict:
    """Get global statistics."""
    print("=== Stats ===")
    resp = requests.get(f"{BASE_URL}/stats", headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()
    print(f"  {data}")
    print()
    return data


def cleanup():
    """Delete all memories for the example entity."""
    print("=== Cleanup ===")
    resp = requests.delete(
        f"{BASE_URL}/memories",
        headers=HEADERS,
        params={"entity_id": ENTITY_ID},
    )
    resp.raise_for_status()
    print(f"  All memories deleted for {ENTITY_ID}.")
    print()


def main():
    print(f"Keyoku Python Client Example")
    print(f"Server: {BASE_URL}\n")

    try:
        # 1. Verify the server is running.
        health_check()

        # 2. Store some memories.
        remember(
            "I'm Alice, a software engineer at Acme Corp. "
            "I prefer dark mode and use TypeScript for most of my projects."
        )
        remember(
            "Alice is working on a dashboard redesign due next Friday. "
            "Her manager Bob approved the new design system."
        )

        # 3. Search by meaning.
        search("What are Alice's preferences?")
        search("project deadlines", mode="important")

        # 4. List all memories.
        memories = list_memories()

        # 5. Run a heartbeat check.
        heartbeat_check()

        # 6. Create a schedule.
        create_schedule("Review weekly metrics", "every_monday_9am")
        list_schedules()

        # 7. Get stats.
        get_stats()

        # 8. Delete a single memory (if any exist).
        if memories:
            delete_memory(memories[0]["id"])

        # 9. Cleanup.
        cleanup()

        print("All examples completed successfully.")

    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to {BASE_URL}")
        print("Make sure keyoku-server is running:")
        print("  ./bin/keyoku-server --db ./memories.db")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
        sys.exit(1)


if __name__ == "__main__":
    main()
