# Troubleshooting Guide

## Common Errors

### "keyoku-server requires KEYOKU_SESSION_TOKEN to be set"

The server refuses to start without a session token. This is a deliberate safety measure -- keyoku-server is designed to be launched by a host application, not run standalone without authentication.

**Fix**: Set the environment variable before starting:

```bash
export KEYOKU_SESSION_TOKEN=$(openssl rand -hex 32)
keyoku-server
```

For development/testing, any non-empty value works:

```bash
KEYOKU_SESSION_TOKEN=dev keyoku-server
```

### "API key is required for provider X"

The configured extraction provider has no API key. The server will fail to initialize if the LLM provider cannot be created.

**Fix**: Set the correct API key for your chosen provider:

| Provider | Required Variable |
|---|---|
| `gemini` (default) | `GEMINI_API_KEY` |
| `openai` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |

If using Gemini for extraction and OpenAI for embeddings, set both `GEMINI_API_KEY` and `OPENAI_API_KEY`.

### "database is locked" / SQLite Busy Errors

SQLite uses a single-writer model. Keyoku configures `MaxOpenConns(1)` and WAL mode with a 5-second busy timeout. If you see lock errors:

**Common causes**:
- Another process has the same SQLite file open (e.g., two keyoku-server instances pointing at the same database).
- A backup tool is holding a lock on the database.
- The WAL file (`keyoku.db-wal`) is corrupted or excessively large.

**Fix**:
1. Ensure only one keyoku-server process uses a given database file.
2. Use `sqlite3 ... ".backup ..."` for backups instead of raw file copies.
3. If WAL is corrupted, stop the server and run: `sqlite3 keyoku.db "PRAGMA wal_checkpoint(TRUNCATE);"`.

### "Embedding dimension mismatch"

This occurs when you switch embedding providers or models after memories have already been stored. The HNSW index is created with fixed dimensions (e.g., 1536 for OpenAI `text-embedding-3-small`, 3072 for Gemini `gemini-embedding-001`). Existing vectors cannot be mixed with vectors of a different size.

**Dimensions by model**:
| Model | Dimensions |
|---|---|
| `text-embedding-3-small` (OpenAI) | 1536 |
| `text-embedding-3-large` (OpenAI) | 3072 |
| `gemini-embedding-001` (Gemini) | 3072 |
| `text-embedding-004` (Gemini) | 768 |

**Fix**: You must re-embed all existing memories if you switch providers. The safest approach:
1. Export memories via `GET /api/v1/memories?entity_id=...`
2. Delete the database and HNSW file.
3. Start with the new embedding config.
4. Re-insert memories via `POST /api/v1/seed`.

### "Memory not found in search"

A memory exists (you can see it via `GET /api/v1/memories`) but does not appear in `POST /api/v1/search` results.

**Common causes**:
1. **Embedding not generated**: If the embedding API call failed during `remember`, the memory is stored without a vector. Search relies on vector similarity and will not find it.
2. **Memory is not in `active` state**: Only active memories are returned by search. Check the memory's `state` field -- it may be `stale`, `archived`, or `deleted`.
3. **Low similarity score**: The default minimum score threshold filters out results below 0.3. Your query may not be semantically close enough.
4. **Wrong entity_id**: Search is scoped to a single entity. Make sure you are querying the correct `entity_id`.

**Debugging steps**:
1. Retrieve the memory directly: `GET /api/v1/memories/<id>`. Check that `state` is `active`.
2. Try a broader search with `min_score: 0.0` to see if it appears with a low score.
3. Check server logs for embedding errors during the original `remember` call.

### "unauthorized" (401)

All endpoints except `/api/v1/health` require a Bearer token matching `KEYOKU_SESSION_TOKEN`.

**Fix**: Include the token in every request:

```bash
curl -H "Authorization: Bearer $KEYOKU_SESSION_TOKEN" \
     http://localhost:18900/api/v1/stats
```

---

## API Error Format

All errors are returned as JSON with an `error` field:

```json
{"error": "description of the problem"}
```

### HTTP Status Codes

| Code | Meaning | Example |
|---|---|---|
| `400` | Bad request -- invalid input | `{"error": "entity_id is required"}`, `{"error": "content is required"}`, `{"error": "invalid request body"}`, `{"error": "query is required"}` |
| `401` | Unauthorized -- missing or invalid Bearer token | `{"error": "unauthorized"}` |
| `404` | Not found | `{"error": "memory not found"}` |
| `500` | Internal server error -- logged server-side, generic message returned | `{"error": "internal server error"}` |

### Validation Errors (400)

The server validates all inputs before processing. Common validation messages:

- `"entity_id is required"` -- missing entity_id field
- `"entity_id contains invalid characters"` -- only `[a-zA-Z0-9_:.\-]` are allowed
- `"entity_id too long (max 256 chars)"` -- ID exceeds length limit
- `"content is required"` -- empty content field
- `"content too large (max 50000 chars)"` -- content exceeds 50KB
- `"content contains invalid UTF-8 encoding"` -- binary data in content field
- `"query is required"` -- search request missing query
- `"invalid memory id"` -- malformed ID in URL path
- `"memories array is required"` -- empty array in seed request

### Internal Errors (500)

When a 500 error occurs, the server logs the real error (`ERROR: ...`) but returns only `{"error": "internal server error"}` to the client. Check the server logs for the actual cause.

---

## Debugging Tips

### Enable Verbose Logging

The server logs all HTTP requests (except health checks) to stdout:

```
2026/03/12 10:15:03 POST /api/v1/remember
2026/03/12 10:15:03 POST /api/v1/search
```

Internal errors are logged with an `ERROR:` prefix:

```
2026/03/12 10:15:04 ERROR: failed to embed: API key invalid
```

To capture logs:

```bash
# Systemd
journalctl -u keyoku-server -f

# Docker
docker logs -f keyoku

# Direct
keyoku-server 2>&1 | tee /var/log/keyoku.log
```

### Check Health Endpoint

The health endpoint is always available without authentication:

```bash
curl http://localhost:18900/api/v1/health
```

Expected response:

```json
{"status": "ok", "timestamp": "2026-03-12T10:15:00Z", "sse_clients": 0}
```

If the server is unreachable, check:
- Is the process running? (`pgrep keyoku-server` or `systemctl status keyoku-server`)
- Is it bound to the expected port? (`ss -tlnp | grep 18900`)
- Is a firewall blocking access?

### Verify SQLite Database

```bash
# Check if the database file exists and has content
ls -la /path/to/keyoku.db*

# Verify database integrity
sqlite3 /path/to/keyoku.db "PRAGMA integrity_check;"

# Check memory count
sqlite3 /path/to/keyoku.db "SELECT COUNT(*) FROM memories;"

# Check WAL mode is active
sqlite3 /path/to/keyoku.db "PRAGMA journal_mode;"
# Expected: wal
```

### Verify Embeddings Are Working

If search returns no results, verify embeddings are being generated:

```bash
# Check if memories have embeddings (non-NULL embedding column)
sqlite3 /path/to/keyoku.db "SELECT id, LENGTH(embedding) FROM memories LIMIT 5;"
```

If `LENGTH(embedding)` is NULL or 0, the embedding API is failing. Check the API key and provider configuration.

### Inspect HNSW Index

The HNSW index file (`keyoku.db.hnsw`) is created alongside the database. If it is missing or corrupted, the index is rebuilt automatically from embedding BLOBs in SQLite on the next startup. This rebuild may take time for large databases.

```bash
# Check if HNSW file exists
ls -la /path/to/keyoku.db.hnsw

# Force a rebuild by deleting it (safe -- server rebuilds on next start)
rm /path/to/keyoku.db.hnsw
# Restart server
```

---

## Performance FAQ

### How many memories can keyoku handle?

Keyoku is tested up to tens of thousands of memories per entity. Practical limits depend on your available RAM for the HNSW index:

| Memories | OpenAI 1536-dim | Gemini 3072-dim |
|---|---|---|
| 1K | ~100 MB | ~200 MB |
| 10K | ~1 GB | ~2 GB |
| 100K | ~10 GB | ~20 GB |

SQLite itself has no practical row limit for this workload. The bottleneck is in-memory HNSW index size.

### What is the expected search latency?

- **Vector search (HNSW)**: Sub-millisecond for under 10K memories. 1-5 ms for 100K memories. The HNSW `EfSearch` parameter (default 50) trades speed for recall.
- **Full search pipeline** (embed query + HNSW + scoring + ranking): 100-500 ms, dominated by the embedding API call over the network.
- **Remember pipeline** (extract + embed + store + dedup + conflict check): 1-3 seconds, dominated by the LLM extraction call.

### When should I tune HNSW parameters?

The defaults (`M=16`, `EfConstruction=200`, `EfSearch=50`) work well for most workloads. Consider tuning if:

- **Recall is too low** (missing relevant results): Increase `EfSearch` (e.g., 100 or 200). This increases search time linearly.
- **Index build is too slow**: Decrease `EfConstruction` (e.g., 100). This trades index quality for build speed.
- **RAM is constrained**: Decrease `M` (e.g., 8 or 12). This reduces memory per node but may hurt recall.

These parameters are set in `vectorindex.DefaultHNSWConfig()` and currently require a code change to modify.

### How does memory decay work?

The background scheduler (enabled by default) runs periodic jobs:

1. **Decay** -- reduces the importance of memories over time based on their `stability` (type-dependent). Memories that decay below the threshold (default 0.3) are marked `stale`.
2. **Archival** -- moves `stale` memories older than 30 days to `archived` state.
3. **Purge** -- deletes `archived` memories older than 90 days.
4. **Consolidation** -- merges groups of similar memories into single consolidated memories using the LLM.

Cron-tagged memories (schedules) are protected from decay.

### SQLite WAL mode considerations

Keyoku opens SQLite in WAL mode with `_busy_timeout=5000` and `_synchronous=NORMAL`. This provides:
- Concurrent reads during writes.
- Crash safety (WAL is replayed on recovery).
- A single-writer guarantee (enforced by `MaxOpenConns(1)`).

The WAL file can grow during heavy write bursts. It is automatically checkpointed by SQLite. If the WAL grows excessively large, you can manually checkpoint:

```bash
sqlite3 /path/to/keyoku.db "PRAGMA wal_checkpoint(TRUNCATE);"
```
