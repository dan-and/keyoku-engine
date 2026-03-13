# Keyoku HTTP API Examples

Curl examples for the keyoku-server HTTP API.

All endpoints are under `/api/v1/`. Default port: `18900`.

## Setup

```bash
# Build and start the server
make build
export OPENAI_API_KEY="sk-..."
export KEYOKU_SESSION_TOKEN="my-secret-token"
./bin/keyoku-server --db ./memories.db
```

Set your token for all requests:

```bash
export TOKEN="my-secret-token"
export BASE="http://localhost:18900/api/v1"
```

---

## Health Check

No authentication required.

```bash
curl -s $BASE/health | jq
```

Response:

```json
{
  "status": "ok",
  "timestamp": "2026-03-12T10:00:00Z",
  "sse_clients": 0
}
```

---

## Remember (Store Memories)

Extract and store memories from content. The engine uses an LLM to extract
individual facts, generate embeddings, and deduplicate automatically.

```bash
curl -s -X POST $BASE/remember \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "entity_id": "user-123",
    "content": "I prefer dark mode and use TypeScript for most projects. My manager is Bob.",
    "agent_id": "assistant-1",
    "source": "chat",
    "session_id": "session-abc"
  }' | jq
```

Response:

```json
{
  "memories_created": 3,
  "memories_updated": 0,
  "memories_deleted": 0,
  "skipped": 0
}
```

Optional fields: `agent_id`, `session_id`, `source`, `schema_id`, `team_id`, `visibility` (one of `private`, `team`, `global`).

---

## Search (Semantic Search)

Find memories by meaning using vector similarity.

```bash
curl -s -X POST $BASE/search \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "entity_id": "user-123",
    "query": "UI preferences",
    "limit": 5,
    "mode": "balanced",
    "min_score": 0.1
  }' | jq
```

Modes: `balanced` (default), `recent`, `important`, `historical`, `comprehensive`.

Response:

```json
[
  {
    "memory": {
      "id": "abc123",
      "entity_id": "user-123",
      "content": "Prefers dark mode",
      "type": "PREFERENCE",
      "importance": 0.7
    },
    "similarity": 0.91,
    "score": 0.87
  }
]
```

For team-aware search (includes team-visible and global memories):

```bash
curl -s -X POST $BASE/search \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "entity_id": "user-123",
    "query": "project status",
    "agent_id": "assistant-1",
    "team_aware": true,
    "limit": 10
  }' | jq
```

---

## List Memories

List all memories for an entity, with optional pagination.

```bash
curl -s "$BASE/memories?entity_id=user-123&limit=10" \
  -H "Authorization: Bearer $TOKEN" | jq
```

---

## Get a Single Memory

```bash
curl -s "$BASE/memories/MEMORY_ID" \
  -H "Authorization: Bearer $TOKEN" | jq
```

---

## Delete a Memory

```bash
curl -s -X DELETE "$BASE/memories/MEMORY_ID" \
  -H "Authorization: Bearer $TOKEN" | jq
```

---

## Delete All Memories for an Entity

```bash
curl -s -X DELETE "$BASE/memories?entity_id=user-123" \
  -H "Authorization: Bearer $TOKEN" | jq
```

---

## Heartbeat Check (Zero-Token)

Fast, local-only scan for actionable signals. No LLM calls.

```bash
curl -s -X POST $BASE/heartbeat/check \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "entity_id": "user-123",
    "deadline_window": "24h",
    "importance_floor": 0.5,
    "max_results": 10,
    "agent_id": "assistant-1"
  }' | jq
```

Response:

```json
{
  "should_act": true,
  "pending_work": [],
  "deadlines": [],
  "scheduled": [],
  "decaying": [],
  "conflicts": [],
  "summary": "1 deadline approaching",
  "priority_action": "Review deadline for project X",
  "urgency": "medium"
}
```

---

## Create a Schedule

Create a recurring task using cron-style tags.

```bash
curl -s -X POST $BASE/schedule \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "entity_id": "user-123",
    "agent_id": "assistant-1",
    "content": "Review weekly metrics and prepare summary",
    "cron_tag": "every_monday_9am"
  }' | jq
```

---

## List Schedules

```bash
curl -s "$BASE/scheduled?entity_id=user-123&agent_id=assistant-1" \
  -H "Authorization: Bearer $TOKEN" | jq
```

---

## Acknowledge a Schedule

Mark a scheduled task as executed (prevents re-firing).

```bash
curl -s -X POST $BASE/schedule/ack \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "memory_id": "SCHEDULE_MEMORY_ID"
  }' | jq
```

---

## Update a Schedule

```bash
curl -s -X PUT "$BASE/schedule/SCHEDULE_MEMORY_ID" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "cron_tag": "every_weekday_8am",
    "content": "Updated: Review daily metrics"
  }' | jq
```

---

## Cancel a Schedule

```bash
curl -s -X DELETE "$BASE/schedule/SCHEDULE_MEMORY_ID" \
  -H "Authorization: Bearer $TOKEN" | jq
```

---

## Global Stats

```bash
curl -s "$BASE/stats" \
  -H "Authorization: Bearer $TOKEN" | jq
```

Per-entity stats:

```bash
curl -s "$BASE/stats/user-123" \
  -H "Authorization: Bearer $TOKEN" | jq
```

---

## Trigger Consolidation

Force immediate memory consolidation (merges similar memories via LLM).

```bash
curl -s -X POST $BASE/consolidate \
  -H "Authorization: Bearer $TOKEN" | jq
```

---

## SSE Event Stream

Subscribe to real-time memory events (memories created, updated, deleted, etc.).

```bash
curl -s -N "$BASE/events" \
  -H "Authorization: Bearer $TOKEN"
```

Events are delivered as Server-Sent Events. Each event has a `type` and JSON `data`.
