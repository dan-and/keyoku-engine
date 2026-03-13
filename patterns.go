// SPDX-License-Identifier: BSL-1.1
// Copyright (c) 2026 Keyoku. All rights reserved.

package keyoku

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/keyoku-ai/keyoku-engine/storage"
)

// BehavioralPattern represents a recurring topic pattern detected on specific days.
type BehavioralPattern struct {
	Description string
	Confidence  float64
	DayOfWeek   *int // 0=Sunday, 6=Saturday; nil if not day-specific
	Topics      []string
	MemoryIDs   []string
}

// detectBehavioralPatterns analyzes memory creation patterns over the last 90 days
// and returns patterns that match today's day of week.
// Uses memory type + tags (LLM-assigned metadata) for topic grouping — no hardcoded keywords.
func (k *Keyoku) detectBehavioralPatterns(ctx context.Context, entityID string) []BehavioralPattern {
	memories, err := k.store.QueryMemories(ctx, storage.MemoryQuery{
		EntityID:   entityID,
		States:     []storage.MemoryState{storage.StateActive, storage.StateStale},
		Limit:      500,
		OrderBy:    "created_at",
		Descending: true,
	})
	if err != nil || len(memories) == 0 {
		return nil
	}

	// Filter to last 90 days
	cutoff := time.Now().AddDate(0, 0, -90)
	var recent []*Memory
	for _, m := range memories {
		if m.CreatedAt.After(cutoff) {
			recent = append(recent, m)
		}
	}
	if len(recent) < 10 {
		return nil // not enough data
	}

	// Group by day-of-week using memory metadata (type + tags) as topic signals.
	// These are assigned by the LLM extraction pipeline, not hardcoded.
	type dayTopic struct {
		day   int
		topic string
	}
	topicCounts := make(map[dayTopic]int)
	topicMemories := make(map[dayTopic][]string)

	for _, m := range recent {
		dow := int(m.CreatedAt.Weekday())
		topics := extractMemoryTopics(m)
		for _, topic := range topics {
			dt := dayTopic{day: dow, topic: topic}
			topicCounts[dt]++
			topicMemories[dt] = append(topicMemories[dt], m.ID)
		}
	}

	// Find patterns: same topic appears 3+ times on same day-of-week
	today := int(time.Now().Weekday())
	var patterns []BehavioralPattern
	seen := make(map[string]bool)

	for dt, count := range topicCounts {
		if count < 3 || dt.day != today {
			continue
		}
		if seen[dt.topic] {
			continue
		}
		seen[dt.topic] = true

		dayName := time.Weekday(dt.day).String()
		confidence := float64(count) / 10.0
		if confidence > 1.0 {
			confidence = 1.0
		}

		dow := dt.day
		patterns = append(patterns, BehavioralPattern{
			Description: fmt.Sprintf("User typically works on %s on %ss", dt.topic, dayName),
			Confidence:  confidence,
			DayOfWeek:   &dow,
			Topics:      []string{dt.topic},
			MemoryIDs:   topicMemories[dt],
		})
	}

	return patterns
}

// extractMemoryTopics derives topic labels from a memory's LLM-assigned metadata.
// Uses memory type (assigned by extraction LLM) and tags (also LLM-assigned) —
// no hardcoded keyword matching.
func extractMemoryTopics(m *Memory) []string {
	var topics []string

	// Memory type is assigned by the LLM extraction pipeline
	topics = append(topics, strings.ToLower(string(m.Type)))

	// Tags are also LLM-assigned during extraction; skip schedule tags
	for _, tag := range m.Tags {
		if strings.HasPrefix(tag, "cron:") {
			continue
		}
		topics = append(topics, strings.ToLower(tag))
	}

	return topics
}
