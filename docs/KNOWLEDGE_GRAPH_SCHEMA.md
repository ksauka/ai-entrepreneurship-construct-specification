# Knowledge Graph Schema

The ETV_V2 graph is a query-aware theory-elaboration graph. Its purpose is to make construct specification, convergence, divergence, and contrast visible across a bibliometric corpus.

## Nodes

- `Publication`
- `Author`
- `Journal`
- `Year`
- `SearchQuery`
- `Topic`
- `VOSCluster`
- `SpecificationProfile`
- `AIRole`
- `AIType`
- `Mechanism`
- `LevelOfAnalysis`
- `ProcessStage`
- `ScopeCondition`
- `DefinitionClarity`
- `SpecificationProblem`

## Relationships

```text
(:Author)-[:WROTE]->(:Publication)
(:Publication)-[:PUBLISHED_IN]->(:Journal)
(:Publication)-[:PUBLISHED_IN_YEAR]->(:Year)
(:Publication)-[:CAPTURED_BY]->(:SearchQuery)
(:Publication)-[:HAS_TOPIC]->(:Topic)
(:Publication)-[:IN_VOS_CLUSTER]->(:VOSCluster)
(:Publication)-[:HAS_SPECIFICATION]->(:SpecificationProfile)

(:SpecificationProfile)-[:SPECIFIES_ROLE]->(:AIRole)
(:SpecificationProfile)-[:SPECIFIES_TYPE]->(:AIType)
(:SpecificationProfile)-[:SPECIFIES_MECHANISM]->(:Mechanism)
(:SpecificationProfile)-[:SPECIFIES_LEVEL]->(:LevelOfAnalysis)
(:SpecificationProfile)-[:SPECIFIES_PROCESS]->(:ProcessStage)
(:SpecificationProfile)-[:SPECIFIES_SCOPE]->(:ScopeCondition)
(:SpecificationProfile)-[:HAS_DEFINITION_CLARITY]->(:DefinitionClarity)
(:SpecificationProfile)-[:HAS_SPECIFICATION_PROBLEM]->(:SpecificationProblem)
```

## Topic Handling

For the MVP, BERTopic topic labels, KeyBERT phrases, and extracted keywords can all be represented as `Topic` nodes. The edge should preserve method metadata:

```text
(:Publication)-[:HAS_TOPIC {
  extraction_method: "keybert",
  relevance_score: 0.82,
  keyword_type: "keybert"
}]->(:Topic {label: "machine learning"})
```

## Specification Handling

AI specification values should not be stored as topics. They are theoretical coding variables and should be represented through `SpecificationProfile`.

This is the layer that supports questions such as:

- Which journals treat AI as a tool, actor, capability, context, or method?
- Which topics have high role fragmentation?
- Which papers use the same AI type but different mechanisms?
- Which search query retrieves the most papers with unclear AI definitions?
- Which VOS clusters contain many papers with missing scope conditions?

## Contrast and Convergence

Later graph-building code should be able to create or derive:

- `CONVERGES_WITH` between papers sharing many specification dimensions.
- `CONTRASTS_WITH` between papers sharing one dimension but differing on another.

Example logic:

- convergence: two papers share at least five of seven specification dimensions.
- contrast: two papers share AI type but differ in role or mechanism.
- fragmentation: a topic, journal, author, or query has high entropy across role or mechanism.
- specification failure: a paper lacks mechanism, scope, or definition clarity.
