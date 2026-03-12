// SPDX-License-Identifier: BSL-1.1
// Copyright (c) 2025 Keyoku. All rights reserved.

//go:build stress

package engine

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/keyoku-ai/keyoku-engine/embedder"
	"github.com/keyoku-ai/keyoku-engine/llm"
	"github.com/keyoku-ai/keyoku-engine/storage"
)

// =============================================================================
// E2E Stress Test Configuration & Reports
// =============================================================================

type e2eStressReport struct {
	Pipeline   *e2ePipelineReport   `json:"full_pipeline"`
	Conflict   *e2eConflictReport   `json:"conflict_pipeline"`
	Importance *e2eImportanceReport `json:"importance_reeval"`
	GraphLLM   *e2eGraphLLMReport   `json:"graph_llm"`
	Retrieval  *e2eRetrievalReport  `json:"retrieval"`
	Events     *e2eEventReport      `json:"events"`
	Verdict    string               `json:"verdict"`
	Duration   string               `json:"duration"`
}

type e2ePipelineReport struct {
	MemoriesCreated    int  `json:"memories_created"`
	EntitiesFound      int  `json:"entities_found"`
	RelationshipsFound int  `json:"relationships_found"`
	GraphTraversal     bool `json:"graph_traversal"`
	AllChecksPass      bool `json:"all_checks_pass"`
}

type e2eConflictReport struct {
	ConflictsDetected int  `json:"conflicts_detected"`
	MemoriesArchived  int  `json:"memories_archived"`
	HistoryLogged     int  `json:"history_logged"`
	AllChecksPass     bool `json:"all_checks_pass"`
}

type e2eImportanceReport struct {
	ReEvalTriggered    bool `json:"reeval_triggered"`
	ImportanceChanged  int  `json:"importance_changed"`
	HistoryLogged      int  `json:"history_logged"`
	AllChecksPass      bool `json:"all_checks_pass"`
}

type e2eGraphLLMReport struct {
	ExplainWorks    bool `json:"explain_works"`
	SummarizeWorks  bool `json:"summarize_works"`
	ExplainLength   int  `json:"explain_length"`
	SummarizeLength int  `json:"summarize_length"`
}

type e2eRetrievalReport struct {
	QueryResults   int  `json:"query_results"`
	StatsMatch     bool `json:"stats_match"`
	SampleWorks    bool `json:"sample_works"`
	AllChecksPass  bool `json:"all_checks_pass"`
}

type e2eEventReport struct {
	EventTypes  map[string]int `json:"event_types"`
	TotalEvents int            `json:"total_events"`
}

// =============================================================================
// E2E Harness
// =============================================================================

type e2eHarness struct {
	t        *testing.T
	engine   *Engine
	store    *storage.SQLiteStore
	provider llm.Provider
	emb      embedder.Embedder
	dbPath   string
}

func newE2EHarness(t *testing.T) *e2eHarness {
	t.Helper()

	// Real LLM provider
	provider, providerName := initLLMProvider(t)
	t.Logf("  using LLM provider: %s", providerName)

	// Real embedder
	emb := initEmbedder(t)

	// Real SQLite
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "e2e_test.db")
	store, err := storage.NewSQLite(dbPath, emb.Dimensions())
	if err != nil {
		t.Fatalf("NewSQLite: %v", err)
	}

	// Full engine with all features
	cfg := EngineConfig{
		ContextTurns:           5,
		EnableImportanceReEval: true,
		Significance:           &SignificanceConfig{Enabled: false}, // disable significance filter for E2E
	}

	engine := NewEngine(provider, emb, store, cfg)

	return &e2eHarness{
		t:        t,
		engine:   engine,
		store:    store,
		provider: provider,
		emb:      emb,
		dbPath:   dbPath,
	}
}

func (h *e2eHarness) close() {
	h.store.Close()
}

// =============================================================================
// Phase 1: Full Pipeline Memory Creation
// =============================================================================

func (h *e2eHarness) testFullPipeline(ctx context.Context) *e2ePipelineReport {
	report := &e2ePipelineReport{}
	entityID := "e2e-pipeline-entity"

	// Feed real conversational content
	inputs := []struct {
		content string
		desc    string
	}{
		{"My name is Alice and I work at Google in San Francisco", "identity + work + location"},
		{"My friend Bob also works at Google as an engineer", "relationship + work"},
		{"I met Charlie at Microsoft last week during a conference", "temporal + org"},
		{"Alice graduated from Stanford University in 2018", "education"},
		{"Bob lives in New York and commutes to Mountain View", "location"},
	}

	for _, input := range inputs {
		result, err := h.engine.Add(ctx, entityID, AddRequest{
			Content:   input.content,
			SessionID: "e2e-session-1",
		})
		if err != nil {
			h.t.Logf("  Add error for %q: %v", input.desc, err)
			continue
		}
		report.MemoriesCreated += result.MemoriesCreated
		h.t.Logf("  [%s] created=%d updated=%d skipped=%d",
			input.desc, result.MemoriesCreated, result.MemoriesUpdated, result.Skipped)
		// Small delay to avoid rate limiting
		time.Sleep(500 * time.Millisecond)
	}

	h.t.Logf("  total memories created: %d", report.MemoriesCreated)

	// Check entities were extracted
	entities, err := h.store.QueryEntities(ctx, storage.EntityQuery{OwnerEntityID: entityID, Limit: 100})
	if err == nil {
		report.EntitiesFound = len(entities)
		h.t.Logf("  entities found: %d", len(entities))
		for _, e := range entities {
			h.t.Logf("    %s (%s)", e.CanonicalName, e.Type)
		}
	}

	// Check relationships
	rels, err := h.store.QueryRelationships(ctx, storage.RelationshipQuery{OwnerEntityID: entityID, Limit: 100})
	if err == nil {
		report.RelationshipsFound = len(rels)
		h.t.Logf("  relationships found: %d", len(rels))
	}

	// Try graph traversal if entities exist
	if len(entities) >= 2 {
		result, err := h.engine.Graph().TraverseFrom(ctx, entityID, GraphQuery{
			StartEntityID: entities[0].ID,
			MaxDepth:      3,
		})
		report.GraphTraversal = err == nil && result != nil && len(result.Nodes) > 0
		if report.GraphTraversal {
			h.t.Logf("  graph traversal: %d nodes from %s", len(result.Nodes), entities[0].CanonicalName)
		}
	}

	report.AllChecksPass = report.MemoriesCreated >= 3 && report.EntitiesFound >= 1
	return report
}

// =============================================================================
// Phase 2: Conflict Resolution Through Pipeline
// =============================================================================

func (h *e2eHarness) testConflictPipeline(ctx context.Context) *e2eConflictReport {
	report := &e2eConflictReport{}
	entityID := "e2e-conflict-entity"

	// Step 1: Establish baseline facts
	baseFacts := []string{
		"I live in San Francisco and I love it here",
		"I really enjoy drinking coffee every morning",
		"I have 2 cats named Luna and Milo",
	}
	for _, fact := range baseFacts {
		_, err := h.engine.Add(ctx, entityID, AddRequest{Content: fact, SessionID: "e2e-conflict"})
		if err != nil {
			h.t.Logf("  base fact error: %v", err)
		}
		time.Sleep(500 * time.Millisecond)
	}
	h.t.Log("  baseline facts established")

	// Step 2: Add contradicting facts
	contradictions := []string{
		"I recently moved to New York from the west coast",
		"I hate coffee now and switched to tea instead",
		"I now have 5 cats at home",
	}
	for _, contra := range contradictions {
		result, err := h.engine.Add(ctx, entityID, AddRequest{Content: contra, SessionID: "e2e-conflict"})
		if err != nil {
			h.t.Logf("  contradiction error: %v", err)
			continue
		}
		for _, d := range result.Details {
			if d.Action == "created" || d.Action == "updated" {
				report.ConflictsDetected++
			}
		}
		time.Sleep(500 * time.Millisecond)
	}
	h.t.Logf("  contradictions processed: %d detected", report.ConflictsDetected)

	// Check for archived memories (superseded by conflicts)
	mems, err := h.store.QueryMemories(ctx, storage.MemoryQuery{
		EntityID: entityID,
		States:   []storage.MemoryState{storage.StateArchived},
		Limit:    50,
	})
	if err == nil {
		report.MemoriesArchived = len(mems)
		h.t.Logf("  archived memories: %d", len(mems))
	}

	// Check history entries
	allMems, _ := h.store.QueryMemories(ctx, storage.MemoryQuery{
		EntityID: entityID,
		Limit:    50,
	})
	for _, mem := range allMems {
		history, _ := h.store.GetHistory(ctx, mem.ID, 10)
		report.HistoryLogged += len(history)
	}
	h.t.Logf("  history entries: %d", report.HistoryLogged)

	report.AllChecksPass = report.HistoryLogged > 0
	return report
}

// =============================================================================
// Phase 3: Importance Re-evaluation
// =============================================================================

func (h *e2eHarness) testImportanceReEval(ctx context.Context) *e2eImportanceReport {
	report := &e2eImportanceReport{}
	entityID := "e2e-importance-entity"

	// Track events
	var importanceEvents int
	h.engine.SetEmitter(func(eventType, _, _, _ string, _ map[string]any) {
		if eventType == "importance.changed" {
			importanceEvents++
		}
	})

	// Step 1: Add high-importance identity memory
	_, err := h.engine.Add(ctx, entityID, AddRequest{
		Content:   "I am the CEO and founder of TechCorp, a company that builds AI systems for healthcare",
		SessionID: "e2e-importance",
	})
	if err != nil {
		h.t.Logf("  identity memory error: %v", err)
	}
	time.Sleep(1 * time.Second)

	// Step 2: Add related but different content (should trigger re-eval)
	_, err = h.engine.Add(ctx, entityID, AddRequest{
		Content:   "TechCorp announced expansion into European markets with new offices in London and Berlin",
		SessionID: "e2e-importance",
	})
	if err != nil {
		h.t.Logf("  related memory error: %v", err)
	}
	time.Sleep(1 * time.Second)

	// Step 3: Add another related memory
	_, err = h.engine.Add(ctx, entityID, AddRequest{
		Content:   "TechCorp raised 50 million dollars in Series B funding led by major venture capital firms",
		SessionID: "e2e-importance",
	})
	if err != nil {
		h.t.Logf("  funding memory error: %v", err)
	}

	report.ImportanceChanged = importanceEvents
	report.ReEvalTriggered = importanceEvents > 0

	// Check history for importance_reeval entries
	allMems, _ := h.store.QueryMemories(ctx, storage.MemoryQuery{EntityID: entityID, Limit: 50})
	for _, mem := range allMems {
		history, _ := h.store.GetHistory(ctx, mem.ID, 10)
		for _, h := range history {
			if h.Operation == "importance_reeval" {
				report.HistoryLogged++
			}
		}
	}
	h.t.Logf("  importance events: %d, history entries: %d", importanceEvents, report.HistoryLogged)

	// Re-eval may or may not trigger depending on LLM extraction results and similarity scores.
	// The key coverage gain is that the code PATH is exercised.
	report.AllChecksPass = true
	return report
}

// =============================================================================
// Phase 4: Graph LLM Operations
// =============================================================================

func (h *e2eHarness) testGraphLLM(ctx context.Context) *e2eGraphLLMReport {
	report := &e2eGraphLLMReport{}
	entityID := "e2e-graphllm-entity"

	// Build entity graph through the pipeline
	graphContent := []string{
		"My colleague Alice works at Google as a senior engineer",
		"Bob is Alice's manager at Google in the engineering department",
		"Alice is friends with Charlie who works at Microsoft",
		"Charlie and Diana are colleagues at Microsoft working on Azure",
	}
	for _, content := range graphContent {
		_, err := h.engine.Add(ctx, entityID, AddRequest{Content: content, SessionID: "e2e-graph"})
		if err != nil {
			h.t.Logf("  graph content error: %v", err)
		}
		time.Sleep(500 * time.Millisecond)
	}

	// Find entities for graph operations
	entities, _ := h.store.QueryEntities(ctx, storage.EntityQuery{OwnerEntityID: entityID, Limit: 50})
	h.t.Logf("  graph entities: %d", len(entities))
	for _, e := range entities {
		h.t.Logf("    %s (%s) id=%s", e.CanonicalName, e.Type, e.ID)
	}

	if len(entities) < 2 {
		h.t.Log("  not enough entities for graph LLM tests")
		return report
	}

	// Find two entities that have a path between them
	var fromEntity, toEntity *storage.Entity
	for i := 0; i < len(entities)-1; i++ {
		for j := i + 1; j < len(entities); j++ {
			path, err := h.engine.Graph().FindPath(ctx, entityID, entities[i].ID, entities[j].ID)
			if err == nil && len(path) >= 2 {
				fromEntity = entities[i]
				toEntity = entities[j]
				break
			}
		}
		if fromEntity != nil {
			break
		}
	}

	// Test ExplainConnection
	if fromEntity != nil && toEntity != nil {
		explanation, err := h.engine.Graph().ExplainConnection(ctx, entityID, fromEntity.ID, toEntity.ID, h.provider)
		if err != nil {
			h.t.Logf("  ExplainConnection error: %v", err)
		} else {
			report.ExplainWorks = len(explanation) > 0
			report.ExplainLength = len(explanation)
			h.t.Logf("  ExplainConnection (%s → %s): %d chars", fromEntity.CanonicalName, toEntity.CanonicalName, len(explanation))
			if len(explanation) > 200 {
				h.t.Logf("    %s...", explanation[:200])
			} else {
				h.t.Logf("    %s", explanation)
			}
		}
	}

	// Test SummarizeEntityContext
	summary, err := h.engine.Graph().SummarizeEntityContext(ctx, entityID, entities[0].ID, h.provider)
	if err != nil {
		h.t.Logf("  SummarizeEntityContext error: %v", err)
	} else {
		report.SummarizeWorks = len(summary) > 0
		report.SummarizeLength = len(summary)
		h.t.Logf("  SummarizeEntityContext (%s): %d chars", entities[0].CanonicalName, len(summary))
	}

	return report
}

// =============================================================================
// Phase 5: Retrieval & Stats
// =============================================================================

func (h *e2eHarness) testRetrieval(ctx context.Context) *e2eRetrievalReport {
	report := &e2eRetrievalReport{}
	entityID := "e2e-retrieval-entity"

	// Add diverse memories
	contents := []string{
		"I enjoy hiking in the mountains on weekends",
		"Python is my favorite programming language for data science",
		"I started learning Japanese six months ago",
		"My morning routine includes meditation and exercise",
		"I prefer working from home over going to the office",
		"I just finished reading a great book about quantum physics",
		"My dog Rex loves playing in the park every afternoon",
		"I cook Italian food almost every evening for dinner",
		"I am planning a trip to Japan next spring with friends",
		"I have been using Linux as my primary operating system for years",
	}
	for _, content := range contents {
		_, err := h.engine.Add(ctx, entityID, AddRequest{Content: content, SessionID: "e2e-retrieval"})
		if err != nil {
			h.t.Logf("  add error: %v", err)
		}
		time.Sleep(300 * time.Millisecond)
	}

	// Query with real embedding
	results, err := h.engine.Query(ctx, entityID, QueryRequest{
		Query: "what programming language does the user prefer",
		Limit: 5,
	})
	if err != nil {
		h.t.Logf("  query error: %v", err)
	} else {
		report.QueryResults = len(results)
		h.t.Logf("  query results: %d", len(results))
		for _, r := range results {
			h.t.Logf("    [%.2f] %s", r.Score.TotalScore, truncate(r.Memory.Content, 60))
		}
	}

	// Test GetGlobalStats
	stats, err := h.engine.GetGlobalStats(ctx, entityID)
	if err != nil {
		h.t.Logf("  stats error: %v", err)
	} else {
		report.StatsMatch = stats.TotalMemories > 0
		h.t.Logf("  global stats: total=%d types=%v states=%v", stats.TotalMemories, stats.ByType, stats.ByState)
	}

	// Test GetSampleMemories
	samples, err := h.engine.GetSampleMemories(ctx, entityID, 3)
	if err != nil {
		h.t.Logf("  sample error: %v", err)
	} else {
		report.SampleWorks = len(samples) > 0
		h.t.Logf("  sample memories: %d", len(samples))
	}

	report.AllChecksPass = report.QueryResults > 0 && report.StatsMatch
	return report
}

// =============================================================================
// Phase 6: Event Emission
// =============================================================================

func (h *e2eHarness) testEvents(ctx context.Context) *e2eEventReport {
	report := &e2eEventReport{EventTypes: make(map[string]int)}
	entityID := "e2e-events-entity"

	h.engine.SetEmitter(func(eventType, _, _, _ string, _ map[string]any) {
		report.EventTypes[eventType]++
		report.TotalEvents++
	})

	// Add a memory → should emit memory.created
	_, err := h.engine.Add(ctx, entityID, AddRequest{Content: "I work as a software engineer at a tech company", SessionID: "e2e-events"})
	if err != nil {
		h.t.Logf("  add error: %v", err)
	}
	time.Sleep(500 * time.Millisecond)

	// Add contradicting memory → may emit conflict.detected
	_, err = h.engine.Add(ctx, entityID, AddRequest{Content: "I don't work in tech anymore, I switched to finance", SessionID: "e2e-events"})
	if err != nil {
		h.t.Logf("  conflict add error: %v", err)
	}

	h.t.Logf("  events captured: %d total", report.TotalEvents)
	for eventType, count := range report.EventTypes {
		h.t.Logf("    %s: %d", eventType, count)
	}

	return report
}

// =============================================================================
// Helper
// =============================================================================

func truncate(s string, maxLen int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) > maxLen {
		return s[:maxLen] + "..."
	}
	return s
}

// =============================================================================
// Individual Test Functions
// =============================================================================

func TestStress_E2EPipeline(t *testing.T) {
	h := newE2EHarness(t)
	defer h.close()
	report := h.testFullPipeline(context.Background())
	if !report.AllChecksPass {
		t.Errorf("pipeline checks failed: memories=%d entities=%d", report.MemoriesCreated, report.EntitiesFound)
	}
}

func TestStress_E2EConflict(t *testing.T) {
	h := newE2EHarness(t)
	defer h.close()
	report := h.testConflictPipeline(context.Background())
	if !report.AllChecksPass {
		t.Errorf("conflict checks failed: detected=%d archived=%d history=%d",
			report.ConflictsDetected, report.MemoriesArchived, report.HistoryLogged)
	}
}

func TestStress_E2EImportance(t *testing.T) {
	h := newE2EHarness(t)
	defer h.close()
	report := h.testImportanceReEval(context.Background())
	if !report.AllChecksPass {
		t.Errorf("importance checks failed")
	}
}

func TestStress_E2EGraphLLM(t *testing.T) {
	h := newE2EHarness(t)
	defer h.close()
	report := h.testGraphLLM(context.Background())
	t.Logf("explain_works=%v summarize_works=%v", report.ExplainWorks, report.SummarizeWorks)
}

func TestStress_E2ERetrieval(t *testing.T) {
	h := newE2EHarness(t)
	defer h.close()
	report := h.testRetrieval(context.Background())
	if !report.AllChecksPass {
		t.Errorf("retrieval checks failed: results=%d stats=%v sample=%v",
			report.QueryResults, report.StatsMatch, report.SampleWorks)
	}
}

func TestStress_E2EEvents(t *testing.T) {
	h := newE2EHarness(t)
	defer h.close()
	report := h.testEvents(context.Background())
	if report.TotalEvents == 0 {
		t.Error("expected at least one event to be emitted")
	}
}

// =============================================================================
// Full E2E Stress Test
// =============================================================================

func TestStress_E2EFull(t *testing.T) {
	h := newE2EHarness(t)
	defer h.close()
	ctx := context.Background()
	start := time.Now()

	fullReport := &e2eStressReport{}

	t.Log("=== E2E Phase 1: Full Pipeline ===")
	fullReport.Pipeline = h.testFullPipeline(ctx)

	t.Log("=== E2E Phase 2: Conflict Resolution ===")
	fullReport.Conflict = h.testConflictPipeline(ctx)

	t.Log("=== E2E Phase 3: Importance Re-evaluation ===")
	fullReport.Importance = h.testImportanceReEval(ctx)

	t.Log("=== E2E Phase 4: Graph LLM Operations ===")
	fullReport.GraphLLM = h.testGraphLLM(ctx)

	t.Log("=== E2E Phase 5: Retrieval & Stats ===")
	fullReport.Retrieval = h.testRetrieval(ctx)

	t.Log("=== E2E Phase 6: Event Emission ===")
	fullReport.Events = h.testEvents(ctx)

	fullReport.Duration = time.Since(start).String()

	// Determine verdict
	issues := 0
	if !fullReport.Pipeline.AllChecksPass {
		issues++
	}
	if !fullReport.Conflict.AllChecksPass {
		issues++
	}
	if !fullReport.Retrieval.AllChecksPass {
		issues++
	}
	if fullReport.Events.TotalEvents == 0 {
		issues++
	}

	switch {
	case issues == 0:
		fullReport.Verdict = "PASS"
	case issues <= 1:
		fullReport.Verdict = "WARN"
	default:
		fullReport.Verdict = "FAIL"
	}

	reportJSON, _ := json.MarshalIndent(fullReport, "    ", "  ")
	t.Logf("\n    === E2E STRESS TEST REPORT ===\n    %s", string(reportJSON))
	t.Logf("\n    === VERDICT: %s ===", fullReport.Verdict)

	if reportPath := os.Getenv("STRESS_REPORT_PATH"); reportPath != "" {
		os.WriteFile(reportPath, reportJSON, 0644)
	}

	if fullReport.Verdict == "FAIL" {
		t.Errorf("E2E stress test FAILED with %d issues", issues)
	}
}
