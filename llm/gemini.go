// SPDX-License-Identifier: BSL-1.1
// Copyright (c) 2025 Keyoku. All rights reserved.

package llm

import (
	"context"
	"encoding/json"
	"fmt"

	"google.golang.org/genai"
)

// GeminiProvider implements Provider using Google's Gemini API via the new google.golang.org/genai SDK.
type GeminiProvider struct {
	client *genai.Client
	model  string
}

func NewGeminiProvider(apiKey, model string) (*GeminiProvider, error) {
	if model == "" {
		model = "gemini-2.5-flash"
	}
	ctx := context.Background()
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create Gemini client: %w", err)
	}
	return &GeminiProvider{client: client, model: model}, nil
}

func (g *GeminiProvider) Name() string  { return "google" }
func (g *GeminiProvider) Model() string { return g.model }

// generate is a helper that calls GenerateContent with the given config and returns the text.
func (g *GeminiProvider) generate(ctx context.Context, prompt string, config *genai.GenerateContentConfig) (string, error) {
	resp, err := g.client.Models.GenerateContent(ctx, g.model, genai.Text(prompt), config)
	if err != nil {
		return "", err
	}
	text := resp.Text()
	if text == "" {
		return "", fmt.Errorf("Gemini returned empty response")
	}
	return text, nil
}

func (g *GeminiProvider) ExtractMemories(ctx context.Context, req ExtractionRequest) (*ExtractionResponse, error) {
	config := &genai.GenerateContentConfig{
		ResponseMIMEType: "application/json",
		ResponseSchema: &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"memories": {
					Type: genai.TypeArray,
					Items: &genai.Schema{
						Type: genai.TypeObject,
						Properties: map[string]*genai.Schema{
							"content":            {Type: genai.TypeString},
							"type":               {Type: genai.TypeString, Enum: []string{"IDENTITY", "PREFERENCE", "RELATIONSHIP", "EVENT", "ACTIVITY", "PLAN", "CONTEXT", "EPHEMERAL"}},
							"importance":         {Type: genai.TypeNumber},
							"confidence":         {Type: genai.TypeNumber},
							"sentiment":          {Type: genai.TypeNumber},
							"importance_factors": {Type: genai.TypeArray, Items: &genai.Schema{Type: genai.TypeString}},
							"confidence_factors": {Type: genai.TypeArray, Items: &genai.Schema{Type: genai.TypeString}},
							"hedging_detected":   {Type: genai.TypeBoolean},
						},
						Required: []string{"content", "type", "importance", "confidence", "sentiment"},
					},
				},
				"entities": {
					Type: genai.TypeArray,
					Items: &genai.Schema{
						Type: genai.TypeObject,
						Properties: map[string]*genai.Schema{
							"canonical_name": {Type: genai.TypeString},
							"type":           {Type: genai.TypeString, Enum: []string{"PERSON", "ORGANIZATION", "LOCATION", "PRODUCT"}},
							"aliases":        {Type: genai.TypeArray, Items: &genai.Schema{Type: genai.TypeString}},
							"context":        {Type: genai.TypeString},
						},
						Required: []string{"canonical_name", "type"},
					},
				},
				"relationships": {
					Type: genai.TypeArray,
					Items: &genai.Schema{
						Type: genai.TypeObject,
						Properties: map[string]*genai.Schema{
							"source":     {Type: genai.TypeString},
							"relation":   {Type: genai.TypeString},
							"target":     {Type: genai.TypeString},
							"confidence": {Type: genai.TypeNumber},
						},
						Required: []string{"source", "relation", "target", "confidence"},
					},
				},
				"updates": {
					Type: genai.TypeArray,
					Items: &genai.Schema{
						Type: genai.TypeObject,
						Properties: map[string]*genai.Schema{
							"query":       {Type: genai.TypeString},
							"new_content": {Type: genai.TypeString},
							"reason":      {Type: genai.TypeString},
						},
						Required: []string{"query", "new_content", "reason"},
					},
				},
				"deletes": {
					Type: genai.TypeArray,
					Items: &genai.Schema{
						Type: genai.TypeObject,
						Properties: map[string]*genai.Schema{
							"query":  {Type: genai.TypeString},
							"reason": {Type: genai.TypeString},
						},
						Required: []string{"query", "reason"},
					},
				},
				"skipped": {
					Type: genai.TypeArray,
					Items: &genai.Schema{
						Type: genai.TypeObject,
						Properties: map[string]*genai.Schema{
							"text":   {Type: genai.TypeString},
							"reason": {Type: genai.TypeString},
						},
						Required: []string{"text", "reason"},
					},
				},
			},
		},
		Temperature: genai.Ptr[float32](0.2),
		TopP:        genai.Ptr[float32](0.8),
	}

	text, err := g.generate(ctx, FormatPrompt(req), config)
	if err != nil {
		return nil, fmt.Errorf("Gemini extraction failed: %w", err)
	}

	var result ExtractionResponse
	if err := json.Unmarshal([]byte(text), &result); err != nil {
		return nil, fmt.Errorf("failed to parse Gemini JSON response: %w", err)
	}
	if err := validateResponse(&result); err != nil {
		return nil, fmt.Errorf("invalid extraction response: %w", err)
	}
	return &result, nil
}

func (g *GeminiProvider) ConsolidateMemories(ctx context.Context, req ConsolidationRequest) (*ConsolidationResponse, error) {
	config := &genai.GenerateContentConfig{
		ResponseMIMEType: "application/json",
		Temperature:      genai.Ptr[float32](0.3),
		TopP:             genai.Ptr[float32](0.8),
	}

	text, err := g.generate(ctx, FormatConsolidationPrompt(req), config)
	if err != nil {
		return nil, fmt.Errorf("Gemini consolidation failed: %w", err)
	}

	var result ConsolidationResponse
	if err := json.Unmarshal([]byte(text), &result); err != nil {
		return nil, fmt.Errorf("failed to parse Gemini consolidation response: %w", err)
	}
	return &result, nil
}

func (g *GeminiProvider) ExtractWithSchema(ctx context.Context, req CustomExtractionRequest) (*CustomExtractionResponse, error) {
	config := &genai.GenerateContentConfig{
		ResponseMIMEType: "application/json",
		ResponseSchema: &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"extracted_data": {Type: genai.TypeObject},
				"confidence":     {Type: genai.TypeNumber},
				"reasoning":      {Type: genai.TypeString},
			},
			Required: []string{"extracted_data", "confidence", "reasoning"},
		},
		Temperature: genai.Ptr[float32](0.2),
		TopP:        genai.Ptr[float32](0.8),
	}

	text, err := g.generate(ctx, FormatCustomExtractionPrompt(req), config)
	if err != nil {
		return nil, fmt.Errorf("Gemini custom extraction failed: %w", err)
	}

	var result CustomExtractionResponse
	if err := json.Unmarshal([]byte(text), &result); err != nil {
		return nil, fmt.Errorf("failed to parse Gemini custom extraction response: %w", err)
	}
	return &result, nil
}

func (g *GeminiProvider) ExtractState(ctx context.Context, req StateExtractionRequest) (*StateExtractionResponse, error) {
	config := &genai.GenerateContentConfig{
		ResponseMIMEType: "application/json",
		ResponseSchema: &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"extracted_state":  {Type: genai.TypeObject},
				"changed_fields":   {Type: genai.TypeArray, Items: &genai.Schema{Type: genai.TypeString}},
				"confidence":       {Type: genai.TypeNumber},
				"reasoning":        {Type: genai.TypeString},
				"suggested_action": {Type: genai.TypeString},
				"validation_error": {Type: genai.TypeString},
			},
			Required: []string{"extracted_state", "changed_fields", "confidence", "reasoning"},
		},
		Temperature: genai.Ptr[float32](0.2),
		TopP:        genai.Ptr[float32](0.8),
	}

	text, err := g.generate(ctx, FormatStateExtractionPrompt(req), config)
	if err != nil {
		return nil, fmt.Errorf("Gemini state extraction failed: %w", err)
	}

	var result StateExtractionResponse
	if err := json.Unmarshal([]byte(text), &result); err != nil {
		return nil, fmt.Errorf("failed to parse Gemini state extraction response: %w", err)
	}
	return &result, nil
}

func (g *GeminiProvider) DetectConflict(ctx context.Context, req ConflictCheckRequest) (*ConflictCheckResponse, error) {
	config := &genai.GenerateContentConfig{
		ResponseMIMEType: "application/json",
		ResponseSchema: &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"contradicts":   {Type: genai.TypeBoolean},
				"conflict_type": {Type: genai.TypeString, Enum: []string{"contradiction", "update", "temporal", "partial", "none"}},
				"confidence":    {Type: genai.TypeNumber},
				"explanation":   {Type: genai.TypeString},
				"resolution":    {Type: genai.TypeString, Enum: []string{"use_new", "keep_existing", "merge", "keep_both"}},
			},
			Required: []string{"contradicts", "conflict_type", "confidence", "explanation", "resolution"},
		},
		Temperature: genai.Ptr[float32](0.2),
		TopP:        genai.Ptr[float32](0.8),
	}

	text, err := g.generate(ctx, FormatConflictCheckPrompt(req), config)
	if err != nil {
		return nil, fmt.Errorf("Gemini conflict check failed: %w", err)
	}

	var result ConflictCheckResponse
	if err := json.Unmarshal([]byte(text), &result); err != nil {
		return nil, fmt.Errorf("failed to parse Gemini conflict check response: %w", err)
	}
	return &result, nil
}

func (g *GeminiProvider) ReEvaluateImportance(ctx context.Context, req ImportanceReEvalRequest) (*ImportanceReEvalResponse, error) {
	config := &genai.GenerateContentConfig{
		ResponseMIMEType: "application/json",
		ResponseSchema: &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"new_importance": {Type: genai.TypeNumber},
				"reason":         {Type: genai.TypeString},
				"should_update":  {Type: genai.TypeBoolean},
			},
			Required: []string{"new_importance", "reason", "should_update"},
		},
		Temperature: genai.Ptr[float32](0.2),
		TopP:        genai.Ptr[float32](0.8),
	}

	text, err := g.generate(ctx, FormatImportanceReEvalPrompt(req), config)
	if err != nil {
		return nil, fmt.Errorf("Gemini importance re-eval failed: %w", err)
	}

	var result ImportanceReEvalResponse
	if err := json.Unmarshal([]byte(text), &result); err != nil {
		return nil, fmt.Errorf("failed to parse Gemini importance re-eval response: %w", err)
	}
	return &result, nil
}

func (g *GeminiProvider) PrioritizeActions(ctx context.Context, req ActionPriorityRequest) (*ActionPriorityResponse, error) {
	config := &genai.GenerateContentConfig{
		ResponseMIMEType: "application/json",
		ResponseSchema: &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"priority_action": {Type: genai.TypeString},
				"action_items":    {Type: genai.TypeArray, Items: &genai.Schema{Type: genai.TypeString}},
				"reasoning":       {Type: genai.TypeString},
				"urgency":         {Type: genai.TypeString, Enum: []string{"immediate", "soon", "can_wait"}},
			},
			Required: []string{"priority_action", "action_items", "reasoning", "urgency"},
		},
		Temperature: genai.Ptr[float32](0.3),
		TopP:        genai.Ptr[float32](0.8),
	}

	text, err := g.generate(ctx, FormatActionPriorityPrompt(req), config)
	if err != nil {
		return nil, fmt.Errorf("Gemini action priority failed: %w", err)
	}

	var result ActionPriorityResponse
	if err := json.Unmarshal([]byte(text), &result); err != nil {
		return nil, fmt.Errorf("failed to parse Gemini action priority response: %w", err)
	}
	return &result, nil
}

func (g *GeminiProvider) AnalyzeHeartbeatContext(ctx context.Context, req HeartbeatAnalysisRequest) (*HeartbeatAnalysisResponse, error) {
	config := &genai.GenerateContentConfig{
		ResponseMIMEType: "application/json",
		ResponseSchema: &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"should_act":          {Type: genai.TypeBoolean},
				"action_brief":        {Type: genai.TypeString},
				"recommended_actions": {Type: genai.TypeArray, Items: &genai.Schema{Type: genai.TypeString}},
				"urgency":             {Type: genai.TypeString, Enum: []string{"none", "low", "medium", "high", "critical"}},
				"reasoning":           {Type: genai.TypeString},
				"autonomy":            {Type: genai.TypeString, Enum: []string{"observe", "suggest", "act"}},
				"user_facing":         {Type: genai.TypeString},
			},
			Required: []string{"should_act", "action_brief", "recommended_actions", "urgency", "reasoning", "autonomy", "user_facing"},
		},
		Temperature: genai.Ptr[float32](0.3),
		TopP:        genai.Ptr[float32](0.8),
	}

	text, err := g.generate(ctx, FormatHeartbeatAnalysisPrompt(req), config)
	if err != nil {
		return nil, fmt.Errorf("Gemini heartbeat analysis failed: %w", err)
	}

	var result HeartbeatAnalysisResponse
	if err := json.Unmarshal([]byte(text), &result); err != nil {
		return nil, fmt.Errorf("failed to parse Gemini heartbeat analysis response: %w", err)
	}
	return &result, nil
}

func (g *GeminiProvider) SummarizeGraph(ctx context.Context, req GraphSummaryRequest) (*GraphSummaryResponse, error) {
	config := &genai.GenerateContentConfig{
		ResponseMIMEType: "application/json",
		ResponseSchema: &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"summary":    {Type: genai.TypeString},
				"confidence": {Type: genai.TypeNumber},
			},
			Required: []string{"summary", "confidence"},
		},
		Temperature: genai.Ptr[float32](0.3),
		TopP:        genai.Ptr[float32](0.8),
	}

	text, err := g.generate(ctx, FormatGraphSummaryPrompt(req), config)
	if err != nil {
		return nil, fmt.Errorf("Gemini graph summary failed: %w", err)
	}

	var result GraphSummaryResponse
	if err := json.Unmarshal([]byte(text), &result); err != nil {
		return nil, fmt.Errorf("failed to parse Gemini graph summary response: %w", err)
	}
	return &result, nil
}

func (g *GeminiProvider) RerankMemories(ctx context.Context, req RerankRequest) (*RerankResponse, error) {
	config := &genai.GenerateContentConfig{
		ResponseMIMEType: "application/json",
		ResponseSchema: &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"rankings": {
					Type: genai.TypeArray,
					Items: &genai.Schema{
						Type: genai.TypeObject,
						Properties: map[string]*genai.Schema{
							"id":    {Type: genai.TypeString},
							"score": {Type: genai.TypeNumber},
						},
						Required: []string{"id", "score"},
					},
				},
			},
			Required: []string{"rankings"},
		},
		Temperature: genai.Ptr[float32](0.1),
		TopP:        genai.Ptr[float32](0.8),
	}

	text, err := g.generate(ctx, FormatRerankPrompt(req), config)
	if err != nil {
		return nil, fmt.Errorf("Gemini rerank failed: %w", err)
	}

	var result RerankResponse
	if err := json.Unmarshal([]byte(text), &result); err != nil {
		return nil, fmt.Errorf("failed to parse Gemini rerank response: %w", err)
	}
	return &result, nil
}
