// Example: Using keyoku-engine as a Go library.
//
// This program demonstrates the core memory operations:
// initializing the engine, storing memories, and searching by meaning.
//
// Prerequisites:
//   - Go 1.24+
//   - An OpenAI API key (or Gemini/Anthropic — see Config options)
//
// Run:
//
//	export OPENAI_API_KEY="sk-..."
//	go run main.go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	keyoku "github.com/keyoku-ai/keyoku-engine"
)

func main() {
	ctx := context.Background()

	// ---------------------------------------------------------------
	// Step 1: Initialize the engine.
	//
	// Config requires a database path (SQLite) and an LLM provider
	// for memory extraction and embeddings. Use ":memory:" for an
	// ephemeral in-memory database, or a file path for persistence.
	// ---------------------------------------------------------------
	k, err := keyoku.New(keyoku.Config{
		DBPath:             "./example-memories.db",
		ExtractionProvider: "openai",
		OpenAIAPIKey:       os.Getenv("OPENAI_API_KEY"),
		EmbeddingModel:     "text-embedding-3-small",
		SchedulerEnabled:   false, // Disable background jobs for this example
	})
	if err != nil {
		log.Fatalf("Failed to initialize keyoku: %v", err)
	}
	defer k.Close()

	// ---------------------------------------------------------------
	// Step 2: Store memories with Remember.
	//
	// Remember() sends content to the LLM for fact extraction, then
	// stores each extracted memory with an embedding for vector search.
	// The entityID groups memories under a user/entity.
	// ---------------------------------------------------------------
	entityID := "user-alice"

	result, err := k.Remember(ctx, entityID,
		"I'm Alice, a software engineer at Acme Corp. I prefer dark mode "+
			"and use TypeScript for most of my projects. My manager is Bob.",
		keyoku.WithSource("onboarding"),
	)
	if err != nil {
		log.Fatalf("Remember failed: %v", err)
	}
	fmt.Printf("First Remember: created=%d, updated=%d, skipped=%d\n",
		result.MemoriesCreated, result.MemoriesUpdated, result.Skipped)

	// Store a second piece of content — the engine will extract new facts
	// and deduplicate against existing memories automatically.
	result, err = k.Remember(ctx, entityID,
		"Alice is working on a dashboard redesign project due next Friday. "+
			"She's been feeling stressed about the tight deadline.",
		keyoku.WithAgentID("assistant-1"),
		keyoku.WithSessionID("session-001"),
	)
	if err != nil {
		log.Fatalf("Remember failed: %v", err)
	}
	fmt.Printf("Second Remember: created=%d, updated=%d, skipped=%d\n",
		result.MemoriesCreated, result.MemoriesUpdated, result.Skipped)

	// ---------------------------------------------------------------
	// Step 3: Search memories by meaning.
	//
	// Search uses vector similarity + scoring to find the most relevant
	// memories. You can control the scoring mode:
	//   - ModeBalanced (default): mix of relevance, recency, importance
	//   - ModeRecent: favor newer memories
	//   - ModeImportant: favor high-importance memories
	//   - ModeHistorical: favor older memories
	//   - ModeComprehensive: cast a wide net
	// ---------------------------------------------------------------
	results, err := k.Search(ctx, entityID, "What are Alice's preferences?",
		keyoku.WithLimit(5),
		keyoku.WithMode(keyoku.ModeBalanced),
	)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	fmt.Printf("\nSearch results for 'What are Alice's preferences?':\n")
	for i, r := range results {
		fmt.Printf("  %d. [score=%.2f] %s (type=%s)\n",
			i+1, r.Score.TotalScore, r.Memory.Content, r.Memory.Type)
	}

	// ---------------------------------------------------------------
	// Step 4: List all memories for an entity.
	// ---------------------------------------------------------------
	memories, err := k.List(ctx, entityID, 20)
	if err != nil {
		log.Fatalf("List failed: %v", err)
	}

	fmt.Printf("\nAll memories for %s (%d total):\n", entityID, len(memories))
	for i, m := range memories {
		fmt.Printf("  %d. [%s] %s (importance=%.1f)\n",
			i+1, m.Type, m.Content, m.Importance)
	}

	// ---------------------------------------------------------------
	// Step 5: Get statistics.
	// ---------------------------------------------------------------
	stats, err := k.Stats(ctx, entityID)
	if err != nil {
		log.Fatalf("Stats failed: %v", err)
	}
	fmt.Printf("\nStats: total_memories=%d\n", stats.TotalMemories)

	// ---------------------------------------------------------------
	// Step 6: Delete a specific memory.
	// ---------------------------------------------------------------
	if len(memories) > 0 {
		id := memories[0].ID
		if err := k.Delete(ctx, id); err != nil {
			log.Fatalf("Delete failed: %v", err)
		}
		fmt.Printf("\nDeleted memory: %s\n", id)
	}

	// ---------------------------------------------------------------
	// Cleanup: remove the example database.
	// ---------------------------------------------------------------
	os.Remove("./example-memories.db")
	os.Remove("./example-memories.db-wal")
	os.Remove("./example-memories.db-shm")
	fmt.Println("\nDone! Example database cleaned up.")
}
