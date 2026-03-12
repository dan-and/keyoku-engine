// SPDX-License-Identifier: BSL-1.1
// Copyright (c) 2025 Keyoku. All rights reserved.

package engine

import "strings"

// expandQueryForFTS generates multiple FTS search queries from a user query.
// This compensates for the semantic gap between how users ask questions
// ("what's the user's name?") and how memories are stored ("User's name is Marcus Chen").
func expandQueryForFTS(query string) []string {
	queries := []string{query}
	lower := strings.ToLower(query)

	// Extract individual meaningful words as additional queries
	// so FTS can match on any keyword in the memory content
	words := strings.Fields(lower)
	skipWords := map[string]bool{
		"what": true, "is": true, "the": true, "a": true, "an": true,
		"user's": true, "user": true, "their": true, "do": true, "does": true,
		"where": true, "how": true, "who": true, "when": true, "which": true,
		"and": true, "or": true, "of": true, "to": true, "in": true,
		"for": true, "with": true, "they": true, "are": true, "was": true,
		"about": true, "from": true, "at": true, "has": true, "have": true,
		"did": true, "can": true, "will": true, "my": true, "me": true,
		"tell": true, "know": true,
	}

	for _, w := range words {
		clean := strings.Trim(w, "?.,!;:'\"")
		if len(clean) > 2 && !skipWords[clean] {
			queries = append(queries, clean)
		}
	}

	// Query intent expansions: map common question patterns to search terms
	// that match how the extraction prompt formats memories
	expansions := []struct {
		patterns []string
		terms    []string
	}{
		{[]string{"name", "called", "who is", "who are"}, []string{"name"}},
		{[]string{"work", "job", "occupation", "employed", "career"}, []string{"works", "occupation", "employed"}},
		{[]string{"live", "location", "where", "reside", "based"}, []string{"lives", "located", "based"}},
		{[]string{"age", "old", "born", "birthday"}, []string{"age", "years old", "born"}},
		{[]string{"school", "education", "university", "college", "degree", "studied"}, []string{"studied", "university", "degree", "graduated"}},
		{[]string{"plan", "future", "going to", "intend", "goal"}, []string{"plans", "intends", "goal", "future"}},
		{[]string{"like", "prefer", "favorite", "favourite", "enjoy"}, []string{"likes", "prefers", "favorite", "enjoys"}},
		{[]string{"friend", "know", "relationship", "family"}, []string{"friend", "knows", "relationship"}},
		{[]string{"boss", "manager", "supervisor", "report to"}, []string{"boss", "manager", "VP", "director", "supervisor", "engineering"}},
		{[]string{"cook", "food", "eat", "cuisine", "recipe", "meal"}, []string{"cooking", "food", "cuisine", "pasta", "recipe"}},
		{[]string{"hobby", "hobbies", "free time", "spare time", "weekend"}, []string{"hobby", "weekend", "free time"}},
		{[]string{"pet", "dog", "cat", "animal"}, []string{"pet", "dog", "cat", "retriever"}},
		{[]string{"invest", "money", "finance", "saving", "retirement"}, []string{"invest", "funds", "retirement", "finance"}},
	}

	for _, exp := range expansions {
		matched := false
		for _, pattern := range exp.patterns {
			if strings.Contains(lower, pattern) {
				matched = true
				break
			}
		}
		if matched {
			for _, term := range exp.terms {
				if !strings.Contains(lower, term) {
					queries = append(queries, term)
				}
			}
		}
	}

	return queries
}

// detectQueryType infers the most likely memory type from a query string.
// Returns the memory type constant or empty string if no strong signal.
func detectQueryType(query string) string {
	lower := strings.ToLower(query)

	typeSignals := []struct {
		memType  string
		keywords []string
	}{
		{"IDENTITY", []string{"name", "who is", "who are", "age", "old", "born", "occupation", "job", "work", "live", "location", "where does", "where do"}},
		{"PREFERENCE", []string{"like", "prefer", "favorite", "favourite", "enjoy", "hate", "dislike", "opinion"}},
		{"RELATIONSHIP", []string{"friend", "family", "knows", "married", "partner", "boss", "colleague", "relationship"}},
		{"PLAN", []string{"plan", "future", "going to", "intend", "goal", "want to", "next"}},
		{"EVENT", []string{"happened", "event", "meeting", "conference", "attended", "trip", "travel"}},
		{"ACTIVITY", []string{"learning", "working on", "currently", "doing", "studying", "practicing"}},
	}

	bestType := ""
	bestScore := 0
	for _, ts := range typeSignals {
		score := 0
		for _, kw := range ts.keywords {
			if strings.Contains(lower, kw) {
				score++
			}
		}
		if score > bestScore {
			bestScore = score
			bestType = ts.memType
		}
	}

	if bestScore > 0 {
		return bestType
	}
	return ""
}
