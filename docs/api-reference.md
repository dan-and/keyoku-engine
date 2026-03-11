# API Reference

All endpoints are under `/api/v1/`. The server listens on port `18900` by default.

Requires `KEYOKU_SESSION_TOKEN` environment variable to be set.

---

## Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Server health check |

---

## Memory

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/remember` | Extract and store memories from content |
| `POST` | `/api/v1/search` | Semantic search across memories |
| `GET` | `/api/v1/memories` | List memories (optionally filtered by entity) |
| `GET` | `/api/v1/memories/:id` | Get a single memory by ID |
| `DELETE` | `/api/v1/memories/:id` | Delete a single memory |
| `DELETE` | `/api/v1/memories` | Delete all memories for an entity |
| `PUT` | `/api/v1/memories/:id/tags` | Update tags on a memory |
| `GET` | `/api/v1/memories/sample` | Get a representative sample of memories |

### POST /api/v1/remember

Extract facts from content and store them as memories.

```json
{
  "entity_id": "user-123",
  "content": "I prefer dark mode and use TypeScript",
  "session_id": "session-abc",
  "agent_id": "agent-1",
  "source": "chat",
  "schema_id": "custom-schema-id",
  "team_id": "team-1",
  "visibility": "private"
}
```

**Response:**
```json
{
  "memories_created": 2,
  "memories_updated": 0,
  "memories_deleted": 0,
  "skipped": 0
}
```

### POST /api/v1/search

Semantic search across stored memories.

```json
{
  "entity_id": "user-123",
  "query": "UI preferences",
  "limit": 5,
  "mode": "balanced",
  "min_score": 0.1,
  "agent_id": "agent-1",
  "team_aware": true
}
```

**Modes:** `balanced`, `recent`, `important`, `historical`, `comprehensive`

**Response:**
```json
[
  {
    "memory": { "id": "...", "content": "Prefers dark mode", ... },
    "similarity": 0.91,
    "score": 0.87
  }
]
```

---

## Heartbeat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/heartbeat/check` | Zero-token heartbeat check (no LLM calls) |
| `POST` | `/api/v1/heartbeat/context` | Combined heartbeat + semantic search + optional LLM analysis |

### POST /api/v1/heartbeat/check

Fast, zero-token check for actionable signals. No LLM or embedding calls.

```json
{
  "entity_id": "user-123",
  "deadline_window": "24h",
  "decay_threshold": 0.3,
  "importance_floor": 0.5,
  "max_results": 10,
  "agent_id": "agent-1",
  "team_id": "team-1"
}
```

**Response:**
```json
{
  "should_act": true,
  "pending_work": [],
  "deadlines": [],
  "scheduled": [],
  "decaying": [],
  "conflicts": [],
  "stale_monitors": [],
  "summary": "1 deadline approaching",
  "priority_action": "Review deadline for project X",
  "action_items": ["..."],
  "urgency": "medium"
}
```

### POST /api/v1/heartbeat/context

Combined endpoint returning heartbeat signals, relevant memories, and optional LLM analysis in a single call.

```json
{
  "entity_id": "user-123",
  "query": "working on the dashboard redesign",
  "top_k": 5,
  "min_score": 0.1,
  "deadline_window": "24h",
  "max_results": 10,
  "agent_id": "agent-1",
  "analyze": true,
  "activity_summary": "User is asking about the dashboard",
  "autonomy": "suggest"
}
```

**Autonomy levels:** `observe` (report only), `suggest` (recommend actions), `act` (take action)

**Response includes:**
- `scheduled`, `deadlines`, `pending_work`, `conflicts` — heartbeat signals
- `relevant_memories` — semantic search results
- `goal_progress` — tracked goals with completion percentage
- `continuity` — session resumption context
- `sentiment_trend` — emotional trajectory
- `relationship_alerts` — entities not mentioned recently
- `knowledge_gaps` — unanswered questions
- `behavioral_patterns` — detected usage patterns
- `analysis` — LLM-generated action recommendations (when `analyze: true`)

---

## Scheduling

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/schedule` | Create a scheduled memory |
| `GET` | `/api/v1/scheduled` | List scheduled memories |
| `POST` | `/api/v1/schedule/ack` | Acknowledge a schedule |
| `PUT` | `/api/v1/schedule/:id` | Update a schedule |
| `DELETE` | `/api/v1/schedule/:id` | Cancel a schedule |

### POST /api/v1/schedule

```json
{
  "entity_id": "user-123",
  "agent_id": "agent-1",
  "content": "Review weekly metrics",
  "cron_tag": "every_monday_9am"
}
```

---

## Watcher

Proactive monitoring with SSE push notifications.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/watcher/start` | Start watching entities |
| `POST` | `/api/v1/watcher/stop` | Stop the watcher |
| `POST` | `/api/v1/watcher/watch` | Add an entity to watch |
| `POST` | `/api/v1/watcher/unwatch` | Remove an entity from watch |

---

## Teams

Multi-agent memory visibility.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/teams` | Create a team |
| `GET` | `/api/v1/teams/:id` | Get team details |
| `DELETE` | `/api/v1/teams/:id` | Delete a team |
| `POST` | `/api/v1/teams/:id/members` | Add agent to team |
| `GET` | `/api/v1/teams/:id/members` | List team members |
| `DELETE` | `/api/v1/teams/:id/members/:agent_id` | Remove agent from team |

---

## Other

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/entities` | List all known entity IDs |
| `GET` | `/api/v1/stats` | Global statistics |
| `GET` | `/api/v1/stats/:entity_id` | Per-entity statistics |
| `POST` | `/api/v1/consolidate` | Trigger immediate memory consolidation |
| `GET` | `/api/v1/events` | SSE stream for real-time memory events |
