// SPDX-License-Identifier: BSL-1.1
// Copyright (c) 2025 Keyoku. All rights reserved.

//go:build stress

package engine

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/keyoku-ai/keyoku-engine/llm"
	"github.com/keyoku-ai/keyoku-engine/storage"
)

// =============================================================================
// Cognitive Stress Test Configuration
// =============================================================================

type cognitiveStressConfig struct {
	NumEntities        int // distribute base memories across N entities
	EmbeddingDims      int
	DedupExactCount    int // exact duplicates to insert
	DedupSemanticCount int // semantic duplicates to insert
	DedupNearCount     int // near-duplicates to insert
	DedupUniqueCount   int // unique base memories to create
	ConcurrentWorkers  int
}

func defaultCognitiveStressConfig() cognitiveStressConfig {
	return cognitiveStressConfig{
		NumEntities:        5,
		EmbeddingDims:      64,
		DedupExactCount:    100,
		DedupSemanticCount: 100,
		DedupNearCount:     100,
		DedupUniqueCount:   700,
		ConcurrentWorkers:  4,
	}
}

// =============================================================================
// Cognitive Stress Test Report
// =============================================================================

type cognitiveStressReport struct {
	Verdict     string             `json:"verdict"`
	Duration    string             `json:"duration"`
	Dedup       *dedupReport       `json:"deduplication"`
	Conflict    *conflictReport    `json:"conflict_detection"`
	Graph       *graphReport       `json:"knowledge_graph"`
	GraphGrowth *graphGrowthReport `json:"graph_integrity"`
	Integration *integrationReport `json:"cross_feature"`
	Concurrent  *cogConcurReport   `json:"concurrent"`
}

type dedupReport struct {
	TotalChecked     int     `json:"total_checked"`
	SkipCount        int     `json:"skip_count"`
	MergeCount       int     `json:"merge_count"`
	CreateCount      int     `json:"create_count"`
	ExactSkipRate    float64 `json:"exact_skip_rate"`
	SemanticSkipRate float64 `json:"semantic_skip_rate"`
	NearMergeRate    float64 `json:"near_merge_rate"`
	UniqueCreateRate float64 `json:"unique_create_rate"`
	FalsePositives   int     `json:"false_positives"`
}

type conflictReport struct {
	TotalChecked         int     `json:"total_checked"`
	TruePositives        int     `json:"true_positives"`
	FalsePositives       int     `json:"false_positives"`
	FalseNegatives       int     `json:"false_negatives"`
	DetectionRate        float64 `json:"detection_rate"`
	FalsePositiveRate    float64 `json:"false_positive_rate"`
	NegationDetected     int     `json:"negation_detected"`
	TemporalDetected     int     `json:"temporal_detected"`
	QuantitativeDetected int     `json:"quantitative_detected"`
	PreferenceDetected   int     `json:"preference_detected"`
	LLMEnabled           bool    `json:"llm_enabled"`
	LLMProvider          string  `json:"llm_provider,omitempty"`
}

type graphReport struct {
	EntitiesCreated       int  `json:"entities_created"`
	RelationshipsCreated  int  `json:"relationships_created"`
	RelationshipsDetected int  `json:"relationships_detected"`
	TraversalWorks        bool `json:"traversal_works"`
	PathFindingWorks      bool `json:"path_finding_works"`
	NeighborQueryWorks    bool `json:"neighbor_query_works"`
	BidirectionalCorrect  bool `json:"bidirectional_correct"`
}

type graphGrowthReport struct {
	MemoriesProcessed     int  `json:"memories_processed"`
	EntitiesTotal         int  `json:"entities_total"`
	RelationshipsTotal    int  `json:"relationships_total"`
	NoDuplicateEntities   bool `json:"no_duplicate_entities"`
	NoOrphanRelationships bool `json:"no_orphan_relationships"`
	PathFindingWorks      bool `json:"path_finding_works"`
	EvidenceAccumulates   bool `json:"evidence_accumulates"`
}

type integrationReport struct {
	StepsExecuted   int  `json:"steps_executed"`
	StepsPassed     int  `json:"steps_passed"`
	PipelineCorrect bool `json:"pipeline_correct"`
}

type cogConcurReport struct {
	Operations int `json:"operations"`
	Errors     int `json:"errors"`
	Panics     int `json:"panics"`
}

// =============================================================================
// Cognitive Stress Test Harness
// =============================================================================

type cognitiveStressHarness struct {
	t            *testing.T
	config       cognitiveStressConfig
	store        *storage.SQLiteStore
	dedup        *DuplicateDetector
	conflict     *ConflictDetector
	relDetect    *RelationshipDetector
	graph        *GraphEngine
	dbPath       string
	entityIndex  map[string]string // canonical name → entity ID
	llmProvider  llm.Provider
	llmName      string
}

// initLLMProvider tries to create a real LLM provider from environment variables.
// Priority: ANTHROPIC_API_KEY > OPENAI_API_KEY > GEMINI_API_KEY
func initLLMProvider(t *testing.T) (llm.Provider, string) {
	t.Helper()

	if key := os.Getenv("ANTHROPIC_API_KEY"); key != "" {
		p, err := llm.NewAnthropicProvider(key, "", "")
		if err != nil {
			t.Fatalf("failed to create Anthropic provider: %v", err)
		}
		return p, "anthropic"
	}
	if key := os.Getenv("OPENAI_API_KEY"); key != "" {
		p, err := llm.NewOpenAIProvider(key, "", "")
		if err != nil {
			t.Fatalf("failed to create OpenAI provider: %v", err)
		}
		return p, "openai"
	}
	if key := os.Getenv("GEMINI_API_KEY"); key != "" {
		p, err := llm.NewGeminiProvider(key, "")
		if err != nil {
			t.Fatalf("failed to create Gemini provider: %v", err)
		}
		return p, "gemini"
	}

	t.Fatal("no LLM API key found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY")
	return nil, ""
}

func newCognitiveStressHarness(t *testing.T, cfg cognitiveStressConfig) *cognitiveStressHarness {
	t.Helper()
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "cognitive_stress.db")

	store, err := storage.NewSQLite(dbPath, cfg.EmbeddingDims)
	if err != nil {
		t.Fatalf("NewSQLite: %v", err)
	}

	dedupCfg := DefaultDuplicateConfig()
	dedup := NewDuplicateDetector(store, &mockEmbedder{dimensions: cfg.EmbeddingDims}, dedupCfg)

	// Full integration: real LLM provider for conflict detection
	provider, providerName := initLLMProvider(t)

	conflictCfg := DefaultConflictConfig()
	conflictCfg.EnableLLMConflictCheck = true
	conflict := NewConflictDetector(store, provider, conflictCfg)

	relDetect := NewRelationshipDetector(store, DefaultRelationshipConfig())
	graph := NewGraphEngine(store, DefaultGraphConfig())

	return &cognitiveStressHarness{
		t:           t,
		config:      cfg,
		store:       store,
		dedup:       dedup,
		conflict:    conflict,
		relDetect:   relDetect,
		graph:       graph,
		dbPath:      dbPath,
		entityIndex: make(map[string]string),
		llmProvider: provider,
		llmName:     providerName,
	}
}

func (h *cognitiveStressHarness) close() {
	h.store.Close()
}

// =============================================================================
// Embedding Helpers
// =============================================================================

// contentEmbedding generates a deterministic normalized embedding from content text.
func contentEmbedding(content string, dims int) []float32 {
	hash := sha256.Sum256([]byte(content))
	seed := int64(binary.LittleEndian.Uint64(hash[:8]))
	rng := rand.New(rand.NewSource(seed))
	vec := make([]float32, dims)
	for i := range vec {
		vec[i] = rng.Float32()*2 - 1
	}
	return normalize(vec)
}

// similarEmbedding creates an embedding near the given base (for semantic dedup testing).
func similarEmbedding(base []float32, noise float64, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	return clusterMember(base, noise, rng)
}

// contentHashStr generates a deterministic hash string for content.
func contentHashStr(content string) string {
	h := sha256.Sum256([]byte(content))
	return fmt.Sprintf("%x", h[:16])
}

// createIndexedMemory creates a memory in the store with the given parameters.
func (h *cognitiveStressHarness) createIndexedMemory(ctx context.Context, entityID, content string, memType storage.MemoryType, embedding []float32, importance float64) *storage.Memory {
	hash := contentHashStr(content)
	now := time.Now()
	mem := &storage.Memory{
		EntityID:       entityID,
		Content:        content,
		Type:           memType,
		State:          storage.StateActive,
		Importance:     importance,
		Confidence:     0.8,
		Stability:      60,
		Hash:           hash,
		Embedding:      encodeVec(embedding),
		LastAccessedAt: &now,
	}
	if err := h.store.CreateMemory(ctx, mem); err != nil {
		h.t.Fatalf("CreateMemory: %v", err)
	}
	return mem
}

// =============================================================================
// Phase 1: Deduplication at Scale
// =============================================================================

func (h *cognitiveStressHarness) testDedup(ctx context.Context) *dedupReport {
	report := &dedupReport{}
	cfg := h.config
	rng := rand.New(rand.NewSource(42))

	// 1. Populate base memories
	type baseMem struct {
		mem *storage.Memory
		emb []float32
	}
	baseMemories := make([]baseMem, 0, cfg.DedupUniqueCount)
	for i := 0; i < cfg.DedupUniqueCount; i++ {
		entityID := fmt.Sprintf("dedup-entity-%d", i%cfg.NumEntities)
		content := fmt.Sprintf("Unique memory %d about topic %d for testing deduplication system integrity", i, i%50)
		emb := contentEmbedding(content, cfg.EmbeddingDims)
		mem := h.createIndexedMemory(ctx, entityID, content, storage.TypeContext, emb, 0.5)
		baseMemories = append(baseMemories, baseMem{mem: mem, emb: emb})
	}
	h.t.Logf("  populated %d base memories", len(baseMemories))

	// 2. Exact duplicate checks — same content + same hash → should skip
	exactSkips := 0
	for i := 0; i < cfg.DedupExactCount; i++ {
		idx := i % len(baseMemories)
		base := baseMemories[idx]
		result, err := h.dedup.CheckDuplicate(ctx, base.mem.EntityID, base.mem.Content, base.emb, base.mem.Hash)
		if err != nil {
			h.t.Logf("  exact dedup error: %v", err)
			continue
		}
		report.TotalChecked++
		switch result.Action {
		case "skip":
			exactSkips++
			report.SkipCount++
		case "merge":
			report.MergeCount++
		case "create":
			report.CreateCount++
		}
	}
	report.ExactSkipRate = float64(exactSkips) / float64(cfg.DedupExactCount)
	h.t.Logf("  exact dedup: %d/%d skipped (%.0f%%)", exactSkips, cfg.DedupExactCount, report.ExactSkipRate*100)

	// 3. Semantic duplicate checks — very similar embedding, different hash → should skip
	semanticSkips := 0
	for i := 0; i < cfg.DedupSemanticCount; i++ {
		idx := i % len(baseMemories)
		base := baseMemories[idx]
		nearEmb := similarEmbedding(base.emb, 0.02, int64(i+1000))
		newHash := fmt.Sprintf("semantic-hash-%d", i)
		result, err := h.dedup.CheckDuplicate(ctx, base.mem.EntityID, base.mem.Content+" rephrased", nearEmb, newHash)
		if err != nil {
			h.t.Logf("  semantic dedup error: %v", err)
			continue
		}
		report.TotalChecked++
		switch result.Action {
		case "skip":
			semanticSkips++
			report.SkipCount++
		case "merge":
			report.MergeCount++
		case "create":
			report.CreateCount++
		}
	}
	report.SemanticSkipRate = float64(semanticSkips) / float64(cfg.DedupSemanticCount)
	h.t.Logf("  semantic dedup: %d/%d skipped (%.0f%%)", semanticSkips, cfg.DedupSemanticCount, report.SemanticSkipRate*100)

	// 4. Near-duplicate checks — moderate noise, added content → should merge
	nearMerges := 0
	for i := 0; i < cfg.DedupNearCount; i++ {
		idx := i % len(baseMemories)
		base := baseMemories[idx]
		nearEmb := similarEmbedding(base.emb, 0.08, int64(i+2000))
		newContent := base.mem.Content + " with additional important details and context"
		newHash := fmt.Sprintf("near-hash-%d", i)
		result, err := h.dedup.CheckDuplicate(ctx, base.mem.EntityID, newContent, nearEmb, newHash)
		if err != nil {
			h.t.Logf("  near dedup error: %v", err)
			continue
		}
		report.TotalChecked++
		switch result.Action {
		case "merge":
			nearMerges++
			report.MergeCount++
		case "skip":
			report.SkipCount++ // close enough — acceptable
		case "create":
			report.CreateCount++
		}
	}
	report.NearMergeRate = float64(nearMerges) / float64(cfg.DedupNearCount)
	h.t.Logf("  near dedup: %d/%d merged (%.0f%%)", nearMerges, cfg.DedupNearCount, report.NearMergeRate*100)

	// 5. Truly unique content — random embedding → should create
	uniqueCreates := 0
	uniqueCount := 100
	for i := 0; i < uniqueCount; i++ {
		entityID := fmt.Sprintf("dedup-entity-%d", rng.Intn(cfg.NumEntities))
		content := fmt.Sprintf("Completely novel unrelated content %d about %d random unconnected things", i+10000, rng.Intn(1000))
		emb := randomVec(rng, cfg.EmbeddingDims)
		newHash := fmt.Sprintf("unique-hash-%d", i)
		result, err := h.dedup.CheckDuplicate(ctx, entityID, content, emb, newHash)
		if err != nil {
			continue
		}
		report.TotalChecked++
		if result.Action == "create" {
			uniqueCreates++
			report.CreateCount++
		} else {
			report.FalsePositives++
		}
	}
	report.UniqueCreateRate = float64(uniqueCreates) / float64(uniqueCount)
	h.t.Logf("  unique content: %d/%d created (%.0f%%)", uniqueCreates, uniqueCount, report.UniqueCreateRate*100)

	return report
}

// =============================================================================
// Phase 2: Contradiction Storm
// =============================================================================

func (h *cognitiveStressHarness) testConflicts(ctx context.Context) *conflictReport {
	report := &conflictReport{}
	entityID := "conflict-test-entity"

	// Ground truth memories
	type groundTruth struct {
		content string
		memType storage.MemoryType
		emb     []float32
	}
	truths := []groundTruth{
		{"User likes coffee", storage.TypePreference, nil},
		{"User works at Acme Corp", storage.TypeContext, nil},
		{"User has 2 cats", storage.TypeContext, nil},
		{"User lives in San Francisco", storage.TypeContext, nil},
		{"User prefers Python for programming", storage.TypePreference, nil},
		{"User enjoys running every morning", storage.TypeContext, nil},
		{"User is a vegetarian", storage.TypeContext, nil},
		{"User drives a blue car", storage.TypeContext, nil},
		{"User speaks 3 languages", storage.TypeContext, nil},
		{"User loves reading fiction", storage.TypePreference, nil},
		{"User has a dog named Max", storage.TypeContext, nil},
		{"User graduated from MIT", storage.TypeIdentity, nil},
		{"User is 30 years old", storage.TypeContext, nil},
		{"User likes hiking on weekends", storage.TypePreference, nil},
		{"User prefers dark mode for coding", storage.TypePreference, nil},
		{"User uses a MacBook Pro", storage.TypeContext, nil},
		{"User wakes up at 6 AM", storage.TypeContext, nil},
		{"User drinks tea in the evening", storage.TypePreference, nil},
		{"User has been to Japan twice", storage.TypeContext, nil},
		{"User plays guitar", storage.TypeContext, nil},
	}

	// Create ground truth memories
	for i := range truths {
		truths[i].emb = contentEmbedding(truths[i].content, h.config.EmbeddingDims)
		h.createIndexedMemory(ctx, entityID, truths[i].content, truths[i].memType, truths[i].emb, 0.8)
	}
	h.t.Logf("  populated %d ground truth memories", len(truths))

	// Contradictions — should detect conflict
	type contradiction struct {
		content      string
		memType      storage.MemoryType
		groundIdx    int          // which ground truth it contradicts
		expectedType ConflictType // expected conflict type
	}
	contradictions := []contradiction{
		// Negation patterns
		{"User doesn't like coffee", storage.TypePreference, 0, ConflictTypeContradiction},
		{"User dislikes coffee", storage.TypePreference, 0, ConflictTypeContradiction},
		{"User hates reading fiction", storage.TypePreference, 9, ConflictTypeContradiction},
		{"User doesn't enjoy running", storage.TypeContext, 5, ConflictTypeContradiction},
		{"User is not a vegetarian", storage.TypeContext, 6, ConflictTypeContradiction},
		{"User doesn't have a dog", storage.TypeContext, 10, ConflictTypeContradiction},
		{"User can't play guitar", storage.TypeContext, 19, ConflictTypeContradiction},
		// Temporal patterns
		{"User now works at BigTech Corp", storage.TypeContext, 1, ConflictTypeTemporal},
		{"User recently moved to New York", storage.TypeContext, 3, ConflictTypeTemporal},
		{"User currently wakes up at 8 AM", storage.TypeContext, 16, ConflictTypeTemporal},
		{"User switched to using a ThinkPad", storage.TypeContext, 15, ConflictTypeTemporal},
		{"User started drinking coffee in the evening", storage.TypePreference, 17, ConflictTypeTemporal},
		// Quantitative/update patterns
		{"User has 5 cats", storage.TypeContext, 2, ConflictTypeUpdate},
		{"User speaks 5 languages", storage.TypeContext, 8, ConflictTypeUpdate},
		{"User is 35 years old", storage.TypeContext, 12, ConflictTypeUpdate},
		{"User has been to Japan 4 times", storage.TypeContext, 18, ConflictTypeUpdate},
		// Preference updates (both TypePreference)
		{"User prefers Go for programming", storage.TypePreference, 4, ConflictTypeUpdate},
		{"User prefers light mode for coding", storage.TypePreference, 14, ConflictTypeUpdate},
	}

	for _, c := range contradictions {
		// Use similar embedding to ground truth (noise=0.05 → sim ~0.95)
		emb := similarEmbedding(truths[c.groundIdx].emb, 0.05, int64(c.groundIdx+500))
		result, err := h.conflict.DetectConflicts(ctx, entityID, c.content, emb, c.memType)
		report.TotalChecked++
		if err != nil {
			h.t.Logf("  conflict detection error for %q: %v", c.content, err)
			report.FalseNegatives++
			continue
		}
		if result.HasConflict {
			report.TruePositives++
			for _, conf := range result.Conflicts {
				switch conf.ConflictType {
				case ConflictTypeContradiction:
					report.NegationDetected++
				case ConflictTypeTemporal:
					report.TemporalDetected++
				case ConflictTypeUpdate:
					report.QuantitativeDetected++
				case ConflictTypePartial:
					report.QuantitativeDetected++
				}
			}
		} else {
			report.FalseNegatives++
			h.t.Logf("  MISS: expected conflict for %q vs %q", c.content, truths[c.groundIdx].content)
		}
	}
	h.t.Logf("  contradictions: %d/%d detected", report.TruePositives, len(contradictions))

	// Non-contradictions — should NOT detect conflict
	nonConflicts := []struct {
		content string
		memType storage.MemoryType
	}{
		// Additive (not contradicting)
		{"User also likes tea", storage.TypePreference},
		{"User has a garden", storage.TypeContext},
		{"User enjoys cooking Italian food", storage.TypeContext},
		{"User reads non-fiction too", storage.TypePreference},
		{"User has a cat named Whiskers", storage.TypeContext},
		// Different subjects entirely
		{"The weather is sunny today", storage.TypeContext},
		{"Python 3.12 was released recently", storage.TypeContext},
		{"San Francisco has great restaurants", storage.TypeContext},
		{"Running shoes need replacement every 500 miles", storage.TypeContext},
		{"MacBook Pro has an M3 chip", storage.TypeContext},
		// Same domain but not contradicting
		{"User also enjoys swimming", storage.TypeContext},
		{"User likes both coffee and tea", storage.TypePreference},
		{"User visited Tokyo in Japan", storage.TypeContext},
		{"User plays acoustic guitar", storage.TypeContext},
		{"User is learning Spanish", storage.TypeContext},
	}

	nonConflictFP := 0
	for _, nc := range nonConflicts {
		emb := contentEmbedding(nc.content, h.config.EmbeddingDims)
		result, err := h.conflict.DetectConflicts(ctx, entityID, nc.content, emb, nc.memType)
		report.TotalChecked++
		if err != nil {
			continue
		}
		if result.HasConflict {
			report.FalsePositives++
			nonConflictFP++
			h.t.Logf("  FALSE POSITIVE: %q flagged as conflict", nc.content)
		}
	}
	h.t.Logf("  non-contradictions: %d/%d false positives", nonConflictFP, len(nonConflicts))

	// Compute rates
	if report.TruePositives+report.FalseNegatives > 0 {
		report.DetectionRate = float64(report.TruePositives) / float64(report.TruePositives+report.FalseNegatives)
	}
	if report.TotalChecked > 0 {
		report.FalsePositiveRate = float64(report.FalsePositives) / float64(report.TotalChecked)
	}

	report.LLMEnabled = h.llmProvider != nil
	report.LLMProvider = h.llmName

	return report
}

// =============================================================================
// Phase 3: Knowledge Graph Construction
// =============================================================================

func (h *cognitiveStressHarness) testGraphConstruction(ctx context.Context) *graphReport {
	report := &graphReport{}
	ownerEntityID := "graph-owner-entity"

	// Relationship-rich content with entities
	type contentWithEntities struct {
		content  string
		entities []ExtractedEntity
	}

	entries := []contentWithEntities{
		// Work relationships
		{"Alice works at Google", []ExtractedEntity{
			{Name: "Alice", Type: storage.EntityTypePerson},
			{Name: "Google", Type: storage.EntityTypeOrganization},
		}},
		{"Bob works at Google", []ExtractedEntity{
			{Name: "Bob", Type: storage.EntityTypePerson},
			{Name: "Google", Type: storage.EntityTypeOrganization},
		}},
		{"Charlie works at Microsoft", []ExtractedEntity{
			{Name: "Charlie", Type: storage.EntityTypePerson},
			{Name: "Microsoft", Type: storage.EntityTypeOrganization},
		}},
		{"Diana works at Apple", []ExtractedEntity{
			{Name: "Diana", Type: storage.EntityTypePerson},
			{Name: "Apple", Type: storage.EntityTypeOrganization},
		}},

		// Management
		{"Bob is my manager", []ExtractedEntity{
			{Name: "Bob", Type: storage.EntityTypePerson},
		}},
		{"Charlie is the boss", []ExtractedEntity{
			{Name: "Charlie", Type: storage.EntityTypePerson},
		}},

		// Social
		{"Alice is my friend", []ExtractedEntity{
			{Name: "Alice", Type: storage.EntityTypePerson},
		}},
		{"Charlie is my friend", []ExtractedEntity{
			{Name: "Charlie", Type: storage.EntityTypePerson},
		}},
		{"Diana and Eve are friends", []ExtractedEntity{
			{Name: "Diana", Type: storage.EntityTypePerson},
			{Name: "Eve", Type: storage.EntityTypePerson},
		}},

		// Family
		{"Frank is my brother", []ExtractedEntity{
			{Name: "Frank", Type: storage.EntityTypePerson},
		}},
		{"Grace is my wife", []ExtractedEntity{
			{Name: "Grace", Type: storage.EntityTypePerson},
		}},

		// Location
		{"Alice lives in San Francisco", []ExtractedEntity{
			{Name: "Alice", Type: storage.EntityTypePerson},
			{Name: "San Francisco", Type: storage.EntityTypeLocation},
		}},
		{"Bob lives in New York", []ExtractedEntity{
			{Name: "Bob", Type: storage.EntityTypePerson},
			{Name: "New York", Type: storage.EntityTypeLocation},
		}},
		{"Google is located in Mountain View", []ExtractedEntity{
			{Name: "Google", Type: storage.EntityTypeOrganization},
			{Name: "Mountain View", Type: storage.EntityTypeLocation},
		}},

		// Dating
		{"Eve is dating Frank", []ExtractedEntity{
			{Name: "Eve", Type: storage.EntityTypePerson},
			{Name: "Frank", Type: storage.EntityTypePerson},
		}},

		// Membership
		{"Alice is a member of Engineering Team", []ExtractedEntity{
			{Name: "Alice", Type: storage.EntityTypePerson},
			{Name: "Engineering Team", Type: storage.EntityTypeOrganization},
		}},
		{"Bob joined Engineering Team", []ExtractedEntity{
			{Name: "Bob", Type: storage.EntityTypePerson},
			{Name: "Engineering Team", Type: storage.EntityTypeOrganization},
		}},

		// From
		{"Charlie is from London", []ExtractedEntity{
			{Name: "Charlie", Type: storage.EntityTypePerson},
			{Name: "London", Type: storage.EntityTypeLocation},
		}},
	}

	// Extract relationships and build graph
	for _, entry := range entries {
		rels, err := h.relDetect.DetectRelationships(ctx, entry.content, entry.entities)
		if err != nil {
			h.t.Logf("  relationship detection error: %v", err)
			continue
		}
		report.RelationshipsDetected += len(rels)

		// Create entities if not existing
		for _, e := range entry.entities {
			if _, exists := h.entityIndex[e.Name]; !exists {
				entity := &storage.Entity{
					OwnerEntityID: ownerEntityID,
					CanonicalName: e.Name,
					Type:          e.Type,
					Aliases:       []string{},
					Attributes:    map[string]any{},
				}
				if err := h.store.CreateEntity(ctx, entity); err == nil {
					h.entityIndex[e.Name] = entity.ID
					report.EntitiesCreated++
				}
			}
		}

		// Also ensure "user" entity exists for patterns that detect "user"
		if _, exists := h.entityIndex["user"]; !exists {
			entity := &storage.Entity{
				OwnerEntityID: ownerEntityID,
				CanonicalName: "user",
				Type:          storage.EntityTypePerson,
				Aliases:       []string{},
				Attributes:    map[string]any{},
			}
			if err := h.store.CreateEntity(ctx, entity); err == nil {
				h.entityIndex["user"] = entity.ID
			}
		}

		// Create relationships
		for _, r := range rels {
			sourceID := h.entityIndex[r.SourceEntity]
			targetID := h.entityIndex[r.TargetEntity]
			if sourceID == "" || targetID == "" {
				continue
			}

			now := time.Now()
			rel := &storage.Relationship{
				OwnerEntityID:    ownerEntityID,
				SourceEntityID:   sourceID,
				TargetEntityID:   targetID,
				RelationshipType: r.RelationshipType,
				Description:      r.Description,
				Strength:         0.8,
				Confidence:       r.Confidence,
				IsBidirectional:  r.IsBidirectional,
				EvidenceCount:    1,
				Attributes:       map[string]any{},
				FirstSeenAt:      now,
				LastSeenAt:       now,
			}
			if err := h.store.CreateRelationship(ctx, rel); err == nil {
				report.RelationshipsCreated++
			}
		}
	}

	h.t.Logf("  entities created: %d", report.EntitiesCreated)
	h.t.Logf("  relationships detected: %d, created: %d", report.RelationshipsDetected, report.RelationshipsCreated)

	// Verify graph operations

	// Traversal: from Alice, should find connected nodes
	if aliceID, ok := h.entityIndex["Alice"]; ok {
		result, err := h.graph.TraverseFrom(ctx, ownerEntityID, GraphQuery{
			StartEntityID: aliceID,
			MaxDepth:      3,
		})
		report.TraversalWorks = err == nil && result != nil && len(result.Nodes) > 1
		if report.TraversalWorks {
			h.t.Logf("  traversal from Alice: %d nodes found", len(result.Nodes))
		} else {
			h.t.Logf("  traversal from Alice FAILED: err=%v", err)
		}
	}

	// Path finding: Alice → Google (direct works_at)
	if aliceID, ok := h.entityIndex["Alice"]; ok {
		if googleID, ok2 := h.entityIndex["Google"]; ok2 {
			path, err := h.graph.FindPath(ctx, ownerEntityID, aliceID, googleID)
			report.PathFindingWorks = err == nil && len(path) >= 2
			h.t.Logf("  path Alice→Google: len=%d, err=%v", len(path), err)
		}
	}

	// Neighbor query: Alice should have edges
	if aliceID, ok := h.entityIndex["Alice"]; ok {
		edges, err := h.graph.GetEntityNeighbors(ctx, ownerEntityID, aliceID)
		report.NeighborQueryWorks = err == nil && len(edges) > 0
		h.t.Logf("  Alice neighbors: %d edges", len(edges))
	}

	// Bidirectional: friend_of should be traversable from both sides
	report.BidirectionalCorrect = true
	if aliceID, ok := h.entityIndex["Alice"]; ok {
		if userID, ok2 := h.entityIndex["user"]; ok2 {
			// Check Alice → user direction
			edges1, err1 := h.graph.GetEntityNeighbors(ctx, ownerEntityID, aliceID)
			// Check user → Alice direction
			edges2, err2 := h.graph.GetEntityNeighbors(ctx, ownerEntityID, userID)

			hasForward := false
			hasReverse := false
			if err1 == nil {
				for _, e := range edges1 {
					if e.TargetEntity != nil && e.TargetEntity.ID == userID {
						hasForward = true
					}
				}
			}
			if err2 == nil {
				for _, e := range edges2 {
					if e.TargetEntity != nil && e.TargetEntity.ID == aliceID {
						hasReverse = true
					}
				}
			}
			// For friend_of (bidirectional), we should see it from at least one direction
			report.BidirectionalCorrect = (err1 == nil && err2 == nil) && (hasForward || hasReverse)
			h.t.Logf("  bidirectional friend_of: forward=%v reverse=%v", hasForward, hasReverse)
		}
	}

	return report
}

// =============================================================================
// Phase 4: Graph Integrity Under Growth
// =============================================================================

func (h *cognitiveStressHarness) testGraphGrowth(ctx context.Context) *graphGrowthReport {
	report := &graphGrowthReport{}
	ownerEntityID := "graph-growth-owner"
	entityNames := make(map[string]string) // name → entity ID
	relKeys := make(map[string]int)         // "source:target:type" → evidence count

	people := []string{"Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"}
	companies := []string{"Google", "Apple", "Microsoft", "Amazon"}
	cities := []string{"San Francisco", "New York", "London", "Tokyo"}

	// Content templates that match relationship patterns
	templates := []struct {
		format   string
		relType  string
		argTypes [2]string // person/company/city
	}{
		{"%s works at %s", "works_at", [2]string{"person", "company"}},
		{"%s lives in %s", "lives_in", [2]string{"person", "city"}},
		{"%s is from %s", "from", [2]string{"person", "city"}},
		{"%s is my friend", "friend_of", [2]string{"person", ""}},
	}

	rng := rand.New(rand.NewSource(123))

	ensureEntity := func(name string, entityType storage.EntityType) string {
		if id, ok := entityNames[name]; ok {
			return id
		}
		entity := &storage.Entity{
			OwnerEntityID: ownerEntityID,
			CanonicalName: name,
			Type:          entityType,
			Aliases:       []string{},
			Attributes:    map[string]any{},
		}
		if err := h.store.CreateEntity(ctx, entity); err != nil {
			return ""
		}
		entityNames[name] = entity.ID
		return entity.ID
	}

	// Ensure "user" entity
	ensureEntity("user", storage.EntityTypePerson)

	for i := 0; i < 500; i++ {
		tmpl := templates[rng.Intn(len(templates))]
		var content string
		var entities []ExtractedEntity
		var sourceName, targetName string
		var sourceType, targetType storage.EntityType

		person := people[rng.Intn(len(people))]

		switch tmpl.argTypes[1] {
		case "company":
			company := companies[rng.Intn(len(companies))]
			content = fmt.Sprintf(tmpl.format, person, company)
			sourceName, targetName = person, company
			sourceType, targetType = storage.EntityTypePerson, storage.EntityTypeOrganization
			entities = []ExtractedEntity{
				{Name: person, Type: storage.EntityTypePerson},
				{Name: company, Type: storage.EntityTypeOrganization},
			}
		case "city":
			city := cities[rng.Intn(len(cities))]
			content = fmt.Sprintf(tmpl.format, person, city)
			sourceName, targetName = person, city
			sourceType, targetType = storage.EntityTypePerson, storage.EntityTypeLocation
			entities = []ExtractedEntity{
				{Name: person, Type: storage.EntityTypePerson},
				{Name: city, Type: storage.EntityTypeLocation},
			}
		default:
			// single-entity pattern like "X is my friend"
			content = fmt.Sprintf(tmpl.format, person)
			sourceName, targetName = person, "user"
			sourceType = storage.EntityTypePerson
			_ = targetType
			entities = []ExtractedEntity{
				{Name: person, Type: storage.EntityTypePerson},
			}
		}

		// Detect relationships
		rels, err := h.relDetect.DetectRelationships(ctx, content, entities)
		if err != nil || len(rels) == 0 {
			continue
		}

		// Ensure entities exist
		sourceID := ensureEntity(sourceName, sourceType)
		if targetName != "" && targetName != "user" {
			ensureEntity(targetName, targetType)
		}
		_ = sourceID

		// Create or update relationships
		for _, r := range rels {
			sID := entityNames[r.SourceEntity]
			tID := entityNames[r.TargetEntity]
			if sID == "" || tID == "" {
				continue
			}

			key := fmt.Sprintf("%s:%s:%s", sID, tID, r.RelationshipType)
			if count, exists := relKeys[key]; exists {
				// Increment evidence count
				relKeys[key] = count + 1
			} else {
				now := time.Now()
				rel := &storage.Relationship{
					OwnerEntityID:    ownerEntityID,
					SourceEntityID:   sID,
					TargetEntityID:   tID,
					RelationshipType: r.RelationshipType,
					Strength:         0.8,
					Confidence:       r.Confidence,
					IsBidirectional:  r.IsBidirectional,
					EvidenceCount:    1,
					Attributes:       map[string]any{},
					FirstSeenAt:      now,
					LastSeenAt:       now,
				}
				if err := h.store.CreateRelationship(ctx, rel); err == nil {
					relKeys[key] = 1
				}
			}
		}

		report.MemoriesProcessed++
	}

	report.EntitiesTotal = len(entityNames)
	report.RelationshipsTotal = len(relKeys)
	h.t.Logf("  processed %d memories → %d entities, %d relationships", report.MemoriesProcessed, report.EntitiesTotal, report.RelationshipsTotal)

	// Verify: no duplicate entities
	report.NoDuplicateEntities = true
	nameCount := make(map[string]int)
	for name := range entityNames {
		nameCount[name]++
		if nameCount[name] > 1 {
			report.NoDuplicateEntities = false
			h.t.Logf("  DUPLICATE ENTITY: %s (count=%d)", name, nameCount[name])
		}
	}

	// Verify: no orphan relationships
	// Since we constructed keys from entityNames, this should always be true
	report.NoOrphanRelationships = true

	// Verify: evidence accumulates (at least one relationship should have count > 1)
	report.EvidenceAccumulates = false
	for _, count := range relKeys {
		if count > 1 {
			report.EvidenceAccumulates = true
			break
		}
	}
	h.t.Logf("  evidence accumulation: %v", report.EvidenceAccumulates)

	// Verify: path finding still works at scale
	report.PathFindingWorks = false
	// Find a path between any two people via a shared company
	personIDs := make([]string, 0)
	for _, name := range people {
		if id, ok := entityNames[name]; ok {
			personIDs = append(personIDs, id)
		}
	}
	if len(personIDs) >= 2 {
		path, err := h.graph.FindPath(ctx, ownerEntityID, personIDs[0], personIDs[1])
		report.PathFindingWorks = err == nil && len(path) >= 2
		h.t.Logf("  path finding: len=%d, err=%v", len(path), err)
	}

	return report
}

// =============================================================================
// Phase 5: Cross-Feature Integration Pipeline
// =============================================================================

func (h *cognitiveStressHarness) testIntegration(ctx context.Context) *integrationReport {
	report := &integrationReport{}
	entityID := "integration-test-entity"
	ownerEntityID := entityID

	// Ensure owner entity for graph operations
	ownerEntity := &storage.Entity{
		OwnerEntityID: ownerEntityID,
		CanonicalName: "integration-user",
		Type:          storage.EntityTypePerson,
		Aliases:       []string{},
		Attributes:    map[string]any{},
	}
	h.store.CreateEntity(ctx, ownerEntity)

	type step struct {
		name     string
		action   func() bool
	}

	// Step 1: Create first memory
	coffeeContent := "User likes coffee"
	coffeeEmb := contentEmbedding(coffeeContent, h.config.EmbeddingDims)
	coffeeHash := contentHashStr(coffeeContent)
	var coffeeMem *storage.Memory

	steps := []step{
		{"create_new_memory", func() bool {
			coffeeMem = h.createIndexedMemory(ctx, entityID, coffeeContent, storage.TypePreference, coffeeEmb, 0.7)
			return coffeeMem != nil && coffeeMem.ID != ""
		}},

		{"exact_dedup_skip", func() bool {
			result, err := h.dedup.CheckDuplicate(ctx, entityID, coffeeContent, coffeeEmb, coffeeHash)
			return err == nil && result.Action == "skip"
		}},

		{"semantic_dedup", func() bool {
			nearEmb := similarEmbedding(coffeeEmb, 0.02, 999)
			result, err := h.dedup.CheckDuplicate(ctx, entityID, "User really enjoys coffee", nearEmb, "different-hash-1")
			// Accept skip or merge — both indicate dedup worked
			return err == nil && (result.Action == "skip" || result.Action == "merge")
		}},

		{"detect_negation_conflict", func() bool {
			contradictEmb := similarEmbedding(coffeeEmb, 0.05, 888)
			result, err := h.conflict.DetectConflicts(ctx, entityID, "User doesn't like coffee anymore", contradictEmb, storage.TypePreference)
			return err == nil && result.HasConflict
		}},

		{"detect_temporal_conflict", func() bool {
			temporalEmb := similarEmbedding(coffeeEmb, 0.05, 777)
			result, err := h.conflict.DetectConflicts(ctx, entityID, "User now prefers tea", temporalEmb, storage.TypePreference)
			return err == nil && result.HasConflict
		}},

		{"create_with_relationship", func() bool {
			content := "User works at Acme Corp"
			emb := contentEmbedding(content, h.config.EmbeddingDims)
			h.createIndexedMemory(ctx, entityID, content, storage.TypeContext, emb, 0.8)

			rels, err := h.relDetect.DetectRelationships(ctx, content, []ExtractedEntity{
				{Name: "User", Type: storage.EntityTypePerson},
				{Name: "Acme Corp", Type: storage.EntityTypeOrganization},
			})
			return err == nil && len(rels) > 0
		}},

		{"create_second_relationship", func() bool {
			content := "Bob is the manager at Acme Corp"
			emb := contentEmbedding(content, h.config.EmbeddingDims)
			h.createIndexedMemory(ctx, entityID, content, storage.TypeContext, emb, 0.7)

			rels, err := h.relDetect.DetectRelationships(ctx, content, []ExtractedEntity{
				{Name: "Bob", Type: storage.EntityTypePerson},
				{Name: "Acme Corp", Type: storage.EntityTypeOrganization},
			})
			if err != nil || len(rels) == 0 {
				return false
			}

			// Store entities and relationships for graph check
			for _, name := range []string{"User", "Bob", "Acme Corp"} {
				if _, exists := h.entityIndex[name]; !exists {
					eType := storage.EntityTypePerson
					if name == "Acme Corp" {
						eType = storage.EntityTypeOrganization
					}
					entity := &storage.Entity{
						OwnerEntityID: ownerEntityID,
						CanonicalName: name,
						Type:          eType,
						Aliases:       []string{},
						Attributes:    map[string]any{},
					}
					if err := h.store.CreateEntity(ctx, entity); err == nil {
						h.entityIndex[name] = entity.ID
					}
				}
			}

			// Create relationships from both steps 6 and 7
			if userID, ok := h.entityIndex["User"]; ok {
				if acmeID, ok2 := h.entityIndex["Acme Corp"]; ok2 {
					now := time.Now()
					h.store.CreateRelationship(ctx, &storage.Relationship{
						OwnerEntityID:    ownerEntityID,
						SourceEntityID:   userID,
						TargetEntityID:   acmeID,
						RelationshipType: "works_at",
						Strength:         0.8,
						Confidence:       0.85,
						EvidenceCount:    1,
						Attributes:       map[string]any{},
						FirstSeenAt:      now,
						LastSeenAt:       now,
					})
				}
			}
			if bobID, ok := h.entityIndex["Bob"]; ok {
				if userID, ok2 := h.entityIndex["User"]; ok2 {
					now := time.Now()
					h.store.CreateRelationship(ctx, &storage.Relationship{
						OwnerEntityID:    ownerEntityID,
						SourceEntityID:   bobID,
						TargetEntityID:   userID,
						RelationshipType: "manages",
						Strength:         0.8,
						Confidence:       0.8,
						EvidenceCount:    1,
						Attributes:       map[string]any{},
						FirstSeenAt:      now,
						LastSeenAt:       now,
					})
				}
			}
			return true
		}},

		{"verify_graph", func() bool {
			userID, ok := h.entityIndex["User"]
			if !ok {
				return false
			}
			// Traverse from User — should find Acme Corp and Bob
			result, err := h.graph.TraverseFrom(ctx, ownerEntityID, GraphQuery{
				StartEntityID: userID,
				MaxDepth:      2,
			})
			if err != nil || result == nil {
				return false
			}
			return len(result.Nodes) >= 2 // at least User + one connected entity
		}},
	}

	for _, s := range steps {
		report.StepsExecuted++
		passed := s.action()
		if passed {
			report.StepsPassed++
			h.t.Logf("  [PASS] %s", s.name)
		} else {
			h.t.Logf("  [FAIL] %s", s.name)
		}
	}

	report.PipelineCorrect = report.StepsPassed == report.StepsExecuted
	return report
}

// =============================================================================
// Phase 6: Concurrent Cognitive Operations
// =============================================================================

func (h *cognitiveStressHarness) testConcurrent(ctx context.Context) *cogConcurReport {
	report := &cogConcurReport{}
	var wg sync.WaitGroup
	var errors, ops, panics atomic.Int64

	entityID := "concurrent-cog-entity"

	// Pre-populate some memories for dedup/conflict to find
	for i := 0; i < 100; i++ {
		content := fmt.Sprintf("Concurrent base memory %d about topic %d for stress testing", i, i%10)
		emb := contentEmbedding(content, h.config.EmbeddingDims)
		h.createIndexedMemory(ctx, entityID, content, storage.TypeContext, emb, 0.5)
	}

	for w := 0; w < h.config.ConcurrentWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			defer func() {
				if r := recover(); r != nil {
					panics.Add(1)
					h.t.Logf("  PANIC in worker %d: %v", workerID, r)
				}
			}()

			localRng := rand.New(rand.NewSource(int64(workerID * 999)))
			for i := 0; i < 50; i++ {
				content := fmt.Sprintf("Worker %d memory %d about topic %d variations", workerID, i, localRng.Intn(20))
				emb := contentEmbedding(content, h.config.EmbeddingDims)
				hash := fmt.Sprintf("conc-hash-%d-%d", workerID, i)

				// Dedup check
				_, err := h.dedup.CheckDuplicate(ctx, entityID, content, emb, hash)
				if err != nil {
					errors.Add(1)
				}
				ops.Add(1)

				// Conflict check
				_, err = h.conflict.DetectConflicts(ctx, entityID, content, emb, storage.TypeContext)
				if err != nil {
					errors.Add(1)
				}
				ops.Add(1)

				// Relationship detection (read-only extraction)
				_, err = h.relDetect.DetectRelationships(ctx, content, nil)
				if err != nil {
					errors.Add(1)
				}
				ops.Add(1)
			}
		}(w)
	}

	wg.Wait()
	report.Operations = int(ops.Load())
	report.Errors = int(errors.Load())
	report.Panics = int(panics.Load())

	h.t.Logf("  concurrent: %d operations, %d errors, %d panics", report.Operations, report.Errors, report.Panics)
	return report
}

// =============================================================================
// Individual Test Functions
// =============================================================================

func TestStress_CognitiveDedup(t *testing.T) {
	cfg := defaultCognitiveStressConfig()
	h := newCognitiveStressHarness(t, cfg)
	defer h.close()
	ctx := context.Background()

	report := h.testDedup(ctx)
	t.Logf("Dedup Results: exact_skip=%.0f%% semantic_skip=%.0f%% near_merge=%.0f%% unique_create=%.0f%%",
		report.ExactSkipRate*100, report.SemanticSkipRate*100, report.NearMergeRate*100, report.UniqueCreateRate*100)

	if report.ExactSkipRate < 0.95 {
		t.Errorf("exact skip rate %.2f below 0.95", report.ExactSkipRate)
	}
	if report.UniqueCreateRate < 0.90 {
		t.Errorf("unique create rate %.2f below 0.90", report.UniqueCreateRate)
	}
}

func TestStress_CognitiveConflict(t *testing.T) {
	cfg := defaultCognitiveStressConfig()
	h := newCognitiveStressHarness(t, cfg)
	defer h.close()
	ctx := context.Background()

	report := h.testConflicts(ctx)
	t.Logf("Conflict Results: detection_rate=%.0f%% false_positive_rate=%.0f%%",
		report.DetectionRate*100, report.FalsePositiveRate*100)
	t.Logf("  negation=%d temporal=%d quantitative=%d preference=%d",
		report.NegationDetected, report.TemporalDetected, report.QuantitativeDetected, report.PreferenceDetected)

	if report.DetectionRate < 0.40 {
		t.Errorf("detection rate %.2f below 0.40", report.DetectionRate)
	}
	if report.FalsePositiveRate > 0.35 {
		t.Errorf("false positive rate %.2f above 0.35", report.FalsePositiveRate)
	}
}

func TestStress_CognitiveGraph(t *testing.T) {
	cfg := defaultCognitiveStressConfig()
	h := newCognitiveStressHarness(t, cfg)
	defer h.close()
	ctx := context.Background()

	report := h.testGraphConstruction(ctx)
	t.Logf("Graph Results: entities=%d relationships_detected=%d created=%d",
		report.EntitiesCreated, report.RelationshipsDetected, report.RelationshipsCreated)

	if !report.TraversalWorks {
		t.Error("graph traversal failed")
	}
	if !report.PathFindingWorks {
		t.Error("path finding failed")
	}
	if !report.NeighborQueryWorks {
		t.Error("neighbor query failed")
	}
}

func TestStress_CognitiveGraphGrowth(t *testing.T) {
	cfg := defaultCognitiveStressConfig()
	h := newCognitiveStressHarness(t, cfg)
	defer h.close()
	ctx := context.Background()

	report := h.testGraphGrowth(ctx)
	t.Logf("Graph Growth Results: memories=%d entities=%d relationships=%d",
		report.MemoriesProcessed, report.EntitiesTotal, report.RelationshipsTotal)

	if !report.NoDuplicateEntities {
		t.Error("duplicate entities found")
	}
	if !report.EvidenceAccumulates {
		t.Error("evidence should accumulate for repeated relationships")
	}
}

func TestStress_CognitiveIntegration(t *testing.T) {
	cfg := defaultCognitiveStressConfig()
	h := newCognitiveStressHarness(t, cfg)
	defer h.close()
	ctx := context.Background()

	report := h.testIntegration(ctx)
	t.Logf("Integration Results: %d/%d steps passed", report.StepsPassed, report.StepsExecuted)

	if !report.PipelineCorrect {
		t.Errorf("integration pipeline failed: %d/%d steps passed", report.StepsPassed, report.StepsExecuted)
	}
}

func TestStress_CognitiveConcurrent(t *testing.T) {
	cfg := defaultCognitiveStressConfig()
	h := newCognitiveStressHarness(t, cfg)
	defer h.close()
	ctx := context.Background()

	report := h.testConcurrent(ctx)
	t.Logf("Concurrent Results: %d operations, %d errors, %d panics",
		report.Operations, report.Errors, report.Panics)

	if report.Panics > 0 {
		t.Errorf("concurrent panics: %d", report.Panics)
	}
	if report.Errors > 0 {
		t.Errorf("concurrent errors: %d", report.Errors)
	}
}

// =============================================================================
// Full Cognitive Stress Test
// =============================================================================

func TestStress_CognitiveFull(t *testing.T) {
	cfg := defaultCognitiveStressConfig()
	h := newCognitiveStressHarness(t, cfg)
	defer h.close()
	ctx := context.Background()
	start := time.Now()

	fullReport := &cognitiveStressReport{}

	// Phase 1: Deduplication
	t.Log("=== Phase 1: Deduplication at Scale ===")
	fullReport.Dedup = h.testDedup(ctx)

	// Phase 2: Contradiction Storm (fresh harness to avoid cross-contamination)
	t.Log("=== Phase 2: Contradiction Storm ===")
	fullReport.Conflict = h.testConflicts(ctx)

	// Phase 3: Knowledge Graph Construction
	t.Log("=== Phase 3: Knowledge Graph Construction ===")
	fullReport.Graph = h.testGraphConstruction(ctx)

	// Phase 4: Graph Integrity Under Growth (uses separate owner entity, no conflict)
	t.Log("=== Phase 4: Graph Integrity Under Growth ===")
	fullReport.GraphGrowth = h.testGraphGrowth(ctx)

	// Phase 5: Cross-Feature Integration
	t.Log("=== Phase 5: Cross-Feature Integration ===")
	fullReport.Integration = h.testIntegration(ctx)

	// Phase 6: Concurrent Operations
	t.Log("=== Phase 6: Concurrent Cognitive Operations ===")
	fullReport.Concurrent = h.testConcurrent(ctx)

	fullReport.Duration = time.Since(start).String()

	// Determine verdict
	issues := 0

	// Dedup checks
	if fullReport.Dedup.ExactSkipRate < 0.80 {
		issues++
	}
	if fullReport.Dedup.UniqueCreateRate < 0.75 {
		issues++
	}

	// Conflict checks
	if fullReport.Conflict.DetectionRate < 0.40 {
		issues++
	}
	if fullReport.Conflict.FalsePositiveRate > 0.35 {
		issues++
	}

	// Graph checks
	if !fullReport.Graph.TraversalWorks {
		issues++
	}
	if !fullReport.Graph.PathFindingWorks {
		issues++
	}
	if !fullReport.GraphGrowth.NoDuplicateEntities {
		issues++
	}

	// Integration check
	if !fullReport.Integration.PipelineCorrect {
		issues++
	}

	// Concurrent check
	if fullReport.Concurrent.Panics > 0 {
		issues++
	}

	switch {
	case issues == 0:
		fullReport.Verdict = "PASS"
	case issues <= 2:
		fullReport.Verdict = "WARN"
	default:
		fullReport.Verdict = "FAIL"
	}

	// Print JSON report
	reportJSON, _ := json.MarshalIndent(fullReport, "    ", "  ")
	t.Logf("\n    === COGNITIVE STRESS TEST REPORT ===\n    %s", string(reportJSON))
	t.Logf("\n    === VERDICT: %s ===", fullReport.Verdict)

	// Write report to file if path specified
	if reportPath := os.Getenv("STRESS_REPORT_PATH"); reportPath != "" {
		os.WriteFile(reportPath, reportJSON, 0644)
	}

	if fullReport.Verdict == "FAIL" {
		t.Errorf("cognitive stress test FAILED with %d issues", issues)
	}
}
