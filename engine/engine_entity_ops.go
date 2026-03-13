// SPDX-License-Identifier: BSL-1.1
// Copyright (c) 2026 Keyoku. All rights reserved.

package engine

import (
	"context"

	"github.com/keyoku-ai/keyoku-engine/llm"
	"github.com/keyoku-ai/keyoku-engine/storage"
)

// extractAndStoreEntities extracts entities from a memory and stores them.
func (e *Engine) extractAndStoreEntities(ctx context.Context, ownerEntityID string, mem *storage.Memory, llmEntities []llm.ExtractedEntity) {
	var extracted []ExtractedEntity
	if len(llmEntities) > 0 {
		for _, le := range llmEntities {
			extracted = append(extracted, ExtractedEntity{
				Name:       le.CanonicalName,
				Type:       mapLLMEntityType(le.Type),
				Context:    le.Context,
				Confidence: 0.9,
			})
		}
	}

	for _, ext := range extracted {
		resolution, err := e.entityResolver.ResolveEntity(ctx, ownerEntityID, ext)
		if err != nil {
			continue
		}

		var entity *storage.Entity

		if resolution.IsNew {
			entity = &storage.Entity{
				OwnerEntityID: ownerEntityID,
				CanonicalName: ext.Name,
				Type:          ext.Type,
				TeamID:        mem.TeamID,
				Aliases:       []string{},
				Attributes:    map[string]any{},
			}

			if e.embedder != nil {
				emb, err := e.embedder.Embed(ctx, ext.Name)
				if err == nil {
					entity.Embedding = encodeEmbedding(emb)
				}
			}

			if err := e.store.CreateEntity(ctx, entity); err != nil {
				continue
			}
		} else {
			entity = resolution.ResolvedEntity
		}

		if entity != nil {
			//nolint:errcheck // fire-and-forget mention tracking
			e.store.CreateEntityMention(ctx, &storage.EntityMention{
				EntityID:       entity.ID,
				MemoryID:       mem.ID,
				MentionText:    ext.Name,
				ContextSnippet: ext.Context,
				Confidence:     ext.Confidence,
			})
		}
	}
}

// detectAndStoreRelationships detects relationships and stores them.
func (e *Engine) detectAndStoreRelationships(ctx context.Context, ownerEntityID string, mem *storage.Memory, llmRelationships []llm.ExtractedRelationship) {
	var detected []DetectedRelationship
	if len(llmRelationships) > 0 {
		for _, lr := range llmRelationships {
			detected = append(detected, DetectedRelationship{
				SourceEntity:     lr.Source,
				TargetEntity:     lr.Target,
				RelationshipType: lr.Relation,
				Confidence:       lr.Confidence,
				Evidence:         mem.Content,
				IsBidirectional:  false,
			})
		}
	}

	for _, rel := range detected {
		sourceEntity, _ := e.store.FindEntityByAlias(ctx, ownerEntityID, rel.SourceEntity)
		targetEntity, _ := e.store.FindEntityByAlias(ctx, ownerEntityID, rel.TargetEntity)

		if sourceEntity == nil || targetEntity == nil {
			continue
		}

		existingRel, _ := e.store.FindRelationship(ctx, ownerEntityID, sourceEntity.ID, targetEntity.ID, rel.RelationshipType)
		if existingRel != nil {
			if rel.Confidence > existingRel.Strength {
				newStrength := (existingRel.Strength + rel.Confidence) / 2
				//nolint:errcheck // fire-and-forget relationship update
				e.store.UpdateRelationship(ctx, existingRel.ID, map[string]any{
					"strength": newStrength,
				})
			}
			//nolint:errcheck // fire-and-forget evidence tracking
			e.store.CreateRelationshipEvidence(ctx, &storage.RelationshipEvidence{
				RelationshipID: existingRel.ID,
				MemoryID:       mem.ID,
				EvidenceText:   rel.Evidence,
				Confidence:     rel.Confidence,
			})
			//nolint:errcheck // fire-and-forget evidence count
			e.store.IncrementRelationshipEvidence(ctx, existingRel.ID)
			continue
		}

		relationship := &storage.Relationship{
			OwnerEntityID:    ownerEntityID,
			SourceEntityID:   sourceEntity.ID,
			TargetEntityID:   targetEntity.ID,
			RelationshipType: rel.RelationshipType,
			TeamID:           mem.TeamID,
			Strength:         rel.Confidence,
			Confidence:       rel.Confidence,
			IsBidirectional:  rel.IsBidirectional,
			Attributes:       map[string]any{},
		}

		if err := e.store.CreateRelationship(ctx, relationship); err != nil {
			continue
		}

		//nolint:errcheck // fire-and-forget evidence tracking
		e.store.CreateRelationshipEvidence(ctx, &storage.RelationshipEvidence{
			RelationshipID: relationship.ID,
			MemoryID:       mem.ID,
			EvidenceText:   rel.Evidence,
			Confidence:     rel.Confidence,
		})
	}
}

// mapLLMEntityType maps LLM entity type strings to storage.EntityType.
func mapLLMEntityType(llmType string) storage.EntityType {
	switch llmType {
	case "PERSON":
		return storage.EntityTypePerson
	case "ORGANIZATION":
		return storage.EntityTypeOrganization
	case "LOCATION":
		return storage.EntityTypeLocation
	case "PRODUCT":
		return storage.EntityTypeProduct
	default:
		return storage.EntityTypeOther
	}
}
