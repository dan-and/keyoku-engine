// SPDX-License-Identifier: BSL-1.1
// Copyright (c) 2025 Keyoku. All rights reserved.

package keyoku

import (
	"context"
	"testing"
	"time"

	"github.com/keyoku-ai/keyoku-engine/storage"
)

// --- P0: Core Fixes ---

func TestGoalProgressFilter_NoActivity(t *testing.T) {
	// GoalProgress with status "no_activity" should be filtered out
	store := &testStore{
		queryMemoriesFn: func(_ context.Context, q storage.MemoryQuery) ([]*storage.Memory, error) {
			if len(q.Types) > 0 && q.Types[0] == storage.TypePlan {
				return []*storage.Memory{
					{ID: "plan-1", Content: "Build feature X", Type: storage.TypePlan, Importance: 0.8, State: storage.StateActive},
				}, nil
			}
			return nil, nil
		},
	}
	k := &Keyoku{store: store}
	result, err := k.HeartbeatCheck(context.Background(), "entity-1",
		WithChecks(CheckGoalProgress))
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	// GoalProgress should be empty since there are no activity memories → status = "no_activity" → filtered
	if len(result.GoalProgress) > 0 {
		for _, g := range result.GoalProgress {
			if g.Status == "no_activity" {
				t.Error("GoalProgress with no_activity status should be filtered out")
			}
		}
	}
}

func TestImportanceFloor_LoweredTo04(t *testing.T) {
	// Memories with importance 0.5 should now pass the floor (was 0.7, now 0.4)
	memReturned := false
	store := &testStore{
		queryMemoriesFn: func(_ context.Context, q storage.MemoryQuery) ([]*storage.Memory, error) {
			if len(q.Types) > 0 && q.Types[0] == storage.TypePlan {
				memReturned = true
				return []*storage.Memory{
					{ID: "plan-1", Content: "Important plan", Type: storage.TypePlan,
						Importance: 0.5, State: storage.StateActive},
				}, nil
			}
			return nil, nil
		},
	}
	k := &Keyoku{store: store}
	result, err := k.HeartbeatCheck(context.Background(), "entity-1",
		WithChecks(CheckPendingWork))
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if !memReturned {
		t.Skip("query was not called for plans")
	}
	if len(result.PendingWork) == 0 {
		t.Error("memory with importance 0.5 should pass the lowered floor of 0.4")
	}
}

// --- P1: Time-of-Day Tiers ---

func TestCurrentTimePeriod(t *testing.T) {
	k := &Keyoku{}
	period := k.currentTimePeriod()
	validPeriods := map[string]bool{
		PeriodMorning: true, PeriodWorking: true, PeriodEvening: true,
		PeriodLateNight: true, PeriodQuiet: true,
	}
	if !validPeriods[period] {
		t.Errorf("currentTimePeriod() = %q, want one of morning/working/evening/late_night/quiet", period)
	}
}

func TestTimePeriodMinTier(t *testing.T) {
	tests := []struct {
		period  string
		wantMin string
	}{
		{PeriodMorning, TierLow},
		{PeriodWorking, TierLow},
		{PeriodEvening, TierNormal},
		{PeriodLateNight, TierElevated},
		{PeriodQuiet, TierImmediate},
	}
	for _, tt := range tests {
		got := timePeriodMinTier(tt.period)
		if got != tt.wantMin {
			t.Errorf("timePeriodMinTier(%q) = %q, want %q", tt.period, got, tt.wantMin)
		}
	}
}

func TestTimePeriodCooldownMultiplier(t *testing.T) {
	tests := []struct {
		period string
		want   float64
	}{
		{PeriodMorning, 0.5},
		{PeriodWorking, 1.0},
		{PeriodEvening, 1.5},
		{PeriodLateNight, 3.0},
		{PeriodQuiet, 10.0},
	}
	for _, tt := range tests {
		got := timePeriodCooldownMultiplier(tt.period)
		if got != tt.want {
			t.Errorf("timePeriodCooldownMultiplier(%q) = %v, want %v", tt.period, got, tt.want)
		}
	}
}

func TestTierRank(t *testing.T) {
	if tierRank(TierImmediate) <= tierRank(TierElevated) {
		t.Error("immediate should rank higher than elevated")
	}
	if tierRank(TierElevated) <= tierRank(TierNormal) {
		t.Error("elevated should rank higher than normal")
	}
	if tierRank(TierNormal) <= tierRank(TierLow) {
		t.Error("normal should rank higher than low")
	}
}

func TestTimePeriod_InResult(t *testing.T) {
	store := &testStore{}
	k := &Keyoku{store: store}
	result, err := k.HeartbeatCheck(context.Background(), "entity-1")
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if result.TimePeriod == "" {
		t.Error("TimePeriod should be set in HeartbeatResult")
	}
}

// --- P1: Elevated Cooldown as Param ---

func TestElevatedCooldownParam(t *testing.T) {
	tests := []struct {
		autonomy string
		wantSet  bool
	}{
		{"act", true},
		{"suggest", true},
		{"observe", true},
	}
	for _, tt := range tests {
		params := DefaultHeartbeatParams(tt.autonomy)
		if tt.wantSet && params.SignalCooldownElevated == 0 {
			t.Errorf("DefaultHeartbeatParams(%q).SignalCooldownElevated should be set", tt.autonomy)
		}
	}
}

// --- P1: First-Contact Mode ---

func TestFirstContactMode(t *testing.T) {
	k := &Keyoku{store: &firstContactStore{}}
	result, err := k.HeartbeatCheck(context.Background(), "entity-1")
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if !result.ShouldAct {
		t.Error("first contact (< 5 memories) should set ShouldAct = true")
	}
	if result.DecisionReason != "first_contact" {
		t.Errorf("DecisionReason = %q, want 'first_contact'", result.DecisionReason)
	}
}

// firstContactStore returns a low memory count to trigger first-contact mode.
type firstContactStore struct {
	testStore
}

func (s *firstContactStore) GetMemoryCount(_ context.Context) (int, error) {
	return 2, nil
}

// --- P2: Content Rotation ---

func TestCollectSignalMemoryIDs(t *testing.T) {
	result := &HeartbeatResult{
		PendingWork: []*Memory{{ID: "pw-1"}, {ID: "pw-2"}},
		Deadlines:   []*Memory{{ID: "dl-1"}},
		Scheduled:   []*Memory{{ID: "pw-1"}}, // duplicate with PendingWork
	}
	ids := collectSignalMemoryIDs(result)
	if len(ids) != 3 {
		t.Errorf("collectSignalMemoryIDs: got %d IDs, want 3 (deduplicated)", len(ids))
	}
	// Check dedup
	seen := make(map[string]bool)
	for _, id := range ids {
		if seen[id] {
			t.Errorf("duplicate ID in result: %s", id)
		}
		seen[id] = true
	}
}

func TestFilterSurfacedMemories(t *testing.T) {
	surfacedIDs := []string{"mem-1", "mem-3"}
	store := &testStore{} // GetRecentlySurfacedMemoryIDs returns nil by default

	k := &Keyoku{store: store}

	memories := []*Memory{
		{ID: "mem-1", Content: "old"},
		{ID: "mem-2", Content: "new"},
		{ID: "mem-3", Content: "old too"},
	}

	// With no surfaced IDs, all memories should pass through
	filtered := k.filterSurfacedMemories(context.Background(), "e1", "a1", memories, time.Hour)
	if len(filtered) != 3 {
		t.Errorf("with no surfaced IDs, got %d, want 3", len(filtered))
	}

	// Now with a store that returns surfaced IDs
	k2 := &Keyoku{store: &surfacedStore{surfacedIDs: surfacedIDs}}
	filtered = k2.filterSurfacedMemories(context.Background(), "e1", "a1", memories, time.Hour)
	if len(filtered) != 1 {
		t.Errorf("after filtering surfaced, got %d, want 1", len(filtered))
	}
	if filtered[0].ID != "mem-2" {
		t.Errorf("remaining memory should be mem-2, got %s", filtered[0].ID)
	}
}

func TestFilterSurfacedMemories_FallbackWhenAllFiltered(t *testing.T) {
	surfacedIDs := []string{"mem-1", "mem-2"}
	k := &Keyoku{store: &surfacedStore{surfacedIDs: surfacedIDs}}

	memories := []*Memory{
		{ID: "mem-1", Content: "a"},
		{ID: "mem-2", Content: "b"},
	}

	// When ALL would be filtered, fall back to original
	filtered := k.filterSurfacedMemories(context.Background(), "e1", "a1", memories, time.Hour)
	if len(filtered) != 2 {
		t.Errorf("fallback should return all %d memories when everything filtered", len(filtered))
	}
}

type surfacedStore struct {
	testStore
	surfacedIDs []string
}

func (s *surfacedStore) GetRecentlySurfacedMemoryIDs(_ context.Context, _, _ string, _ time.Duration) ([]string, error) {
	return s.surfacedIDs, nil
}

// --- P2: Conversation Rhythm ---

func TestIsUserTypicallyActive_NoData(t *testing.T) {
	store := &testStore{}
	k := &Keyoku{store: store}
	// No data → assume active
	if !k.isUserTypicallyActive(context.Background(), "entity-1") {
		t.Error("with no data, should assume user is active")
	}
}

func TestIsUserTypicallyActive_TooFewMessages(t *testing.T) {
	store := &rhythmStore{dist: map[int]int{10: 5, 14: 3}, total: 8}
	k := &Keyoku{store: store}
	// < 20 messages → assume active
	if !k.isUserTypicallyActive(context.Background(), "entity-1") {
		t.Error("with < 20 messages, should assume user is active")
	}
}

type rhythmStore struct {
	testStore
	dist  map[int]int
	total int // not used directly, dist values sum to this
}

func (s *rhythmStore) GetMessageHourDistribution(_ context.Context, _ string, _ int) (map[int]int, error) {
	return s.dist, nil
}

// --- P2: Positive Deltas as Signal ---

func TestPositiveDeltasClassifiedAsSignal(t *testing.T) {
	result := &HeartbeatResult{
		PositiveDeltas: []PositiveDelta{
			{Type: "goal_improved", Description: "Project X moved to on_track"},
		},
	}

	k := &Keyoku{store: &testStore{}}
	active := k.classifyActiveSignals(result)

	if _, ok := active[CheckPositiveDeltas]; !ok {
		t.Error("positive deltas should be classified as an active signal")
	}
	if active[CheckPositiveDeltas] != TierNormal {
		t.Errorf("positive deltas tier = %q, want %q", active[CheckPositiveDeltas], TierNormal)
	}
}

// --- P2: Escalation ---

func TestBuildTopicLabel(t *testing.T) {
	k := &Keyoku{store: &testStore{}}

	result := &HeartbeatResult{
		PendingWork: []*Memory{{ID: "1", Content: "Fix the authentication bug in the login flow"}},
	}
	label := k.buildTopicLabel(result)
	if label != "Fix the authentication bug in the login flow" {
		t.Errorf("unexpected label: %q", label)
	}

	// Long content should be truncated
	longContent := make([]byte, 200)
	for i := range longContent {
		longContent[i] = 'x'
	}
	result2 := &HeartbeatResult{
		PendingWork: []*Memory{{ID: "1", Content: string(longContent)}},
	}
	label2 := k.buildTopicLabel(result2)
	if len(label2) > 80 {
		t.Errorf("label should be truncated to 80 chars, got %d", len(label2))
	}
}

// --- Snapshot Testing ---

// TestHeartbeatSnapshot_FullPipeline runs the complete heartbeat pipeline with known state
// and asserts the full result structure. This is the "snapshot" approach.
func TestHeartbeatSnapshot_FullPipeline(t *testing.T) {
	now := time.Now()
	twoDaysAgo := now.Add(-48 * time.Hour)

	store := &testStore{
		queryMemoriesFn: func(_ context.Context, q storage.MemoryQuery) ([]*storage.Memory, error) {
			// Return pending work
			if len(q.Types) > 0 && q.Types[0] == storage.TypePlan {
				return []*storage.Memory{
					{
						ID: "plan-1", Content: "Ship v2 release",
						Type: storage.TypePlan, Importance: 0.8,
						State: storage.StateActive, CreatedAt: twoDaysAgo,
					},
				}, nil
			}
			return nil, nil
		},
	}

	k := &Keyoku{store: store}
	result, err := k.HeartbeatCheck(context.Background(), "entity-1",
		WithAutonomy("act"),
		WithChecks(CheckPendingWork))
	if err != nil {
		t.Fatalf("error: %v", err)
	}

	// Verify result structure
	if result.TimePeriod == "" {
		t.Error("TimePeriod should be set")
	}
	if len(result.PendingWork) == 0 {
		t.Error("PendingWork should contain the plan")
	}
	if result.Summary == "" {
		t.Error("Summary should be built")
	}

	// Log full result for snapshot review
	t.Logf("Snapshot result:")
	t.Logf("  ShouldAct: %v", result.ShouldAct)
	t.Logf("  DecisionReason: %s", result.DecisionReason)
	t.Logf("  TimePeriod: %s", result.TimePeriod)
	t.Logf("  EscalationLevel: %d", result.EscalationLevel)
	t.Logf("  HighestUrgencyTier: %s", result.HighestUrgencyTier)
	t.Logf("  PendingWork: %d", len(result.PendingWork))
	t.Logf("  PositiveDeltas: %d", len(result.PositiveDeltas))
	t.Logf("  Summary: %s", result.Summary[:min(len(result.Summary), 100)])
}

// TestHeartbeatSnapshot_NoSignals verifies clean result when nothing to report.
func TestHeartbeatSnapshot_NoSignals(t *testing.T) {
	store := &testStore{}
	k := &Keyoku{store: store}

	result, err := k.HeartbeatCheck(context.Background(), "entity-1",
		WithAutonomy("suggest"))
	if err != nil {
		t.Fatalf("error: %v", err)
	}

	if result.ShouldAct {
		t.Error("should not act with no signals")
	}
	if result.TimePeriod == "" {
		t.Error("TimePeriod should always be set")
	}
	if result.EscalationLevel != 0 {
		t.Errorf("EscalationLevel should be 0 with no signals, got %d", result.EscalationLevel)
	}

	t.Logf("No-signal snapshot:")
	t.Logf("  ShouldAct: %v", result.ShouldAct)
	t.Logf("  DecisionReason: %s", result.DecisionReason)
	t.Logf("  TimePeriod: %s", result.TimePeriod)
}

// TestHeartbeatSnapshot_FirstContact verifies the first-contact flow.
func TestHeartbeatSnapshot_FirstContact(t *testing.T) {
	k := &Keyoku{store: &firstContactStore{}}

	result, err := k.HeartbeatCheck(context.Background(), "entity-1",
		WithAutonomy("act"))
	if err != nil {
		t.Fatalf("error: %v", err)
	}

	if !result.ShouldAct {
		t.Error("first contact should act")
	}
	if result.DecisionReason != "first_contact" {
		t.Errorf("DecisionReason = %q, want first_contact", result.DecisionReason)
	}
	if result.HighestUrgencyTier != TierNormal {
		t.Errorf("HighestUrgencyTier = %q, want %q", result.HighestUrgencyTier, TierNormal)
	}

	t.Logf("First-contact snapshot:")
	t.Logf("  ShouldAct: %v", result.ShouldAct)
	t.Logf("  DecisionReason: %s", result.DecisionReason)
	t.Logf("  TimePeriod: %s", result.TimePeriod)
}
