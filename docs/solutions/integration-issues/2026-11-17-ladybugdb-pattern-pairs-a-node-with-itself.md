---
title: "A LadybugDB co-change query pairs a file with itself, and returns more rows than the same SQL in DuckDB"
category: integration-issues
date: 2026-11-17
tags: [ladybugdb, lbug, cypher, duckdb, repograph, walk-semantics, graph]
symptom: "`MATCH (f:File)<-[:CHANGED]-(c:Commit)-[:CHANGED]->(g:File)` returns rows where g is f (75 pairs where DuckDB's self-join returns 60)"
root_cause: "LadybugDB matches a non-recursive pattern with walk semantics: the same relationship may be used twice in one match, so both CHANGED edges can be the one edge c→f. Neo4j uses trail semantics (each relationship at most once per match) and never returns those rows."
---

# A LadybugDB co-change query pairs a file with itself

## Problem

`docs/streeling/graph.cypher` question 4 asks which files change together with a learning's files:

```cypher
MATCH (f:File)<-[:CHANGED]-(c:Commit)-[:CHANGED]->(g:File)
```

Compared with the same question in SQL over the graph's CSV files
(`docs/streeling/graph-baseline.sql`, a self-join of `changed.csv` with `a.file <> b.file`), LadybugDB
returned 75 pairs and DuckDB 60. The 15 extra rows all had `g = f`.

Read with Neo4j's rules, the pattern cannot produce them: in Neo4j a relationship appears at most once per
match, so the two `CHANGED` edges must be two different edges, hence two different files.

## Root cause

LadybugDB (the Kùzu fork, CLI `lbug`) matches non-recursive patterns with **walk** semantics: a relationship
may repeat inside one match. Both `CHANGED` edges of the pattern can bind to the same edge `c→f`, which makes
`g` equal to `f`.

## Fix

Say it in the query: `WHERE g.id <> f.id` (commit 37a5a63b, branch `feat/ixforge-ladybug`). With this
change, and with commit dates in UTC to the second plus the SHA as tie-break, the two engines give the same
rows for questions 1–6 and 8, compared as multisets without `LIMIT`.

Any pattern that goes out and comes back along the same relationship type needs the same guard: co-change,
PRs that touch the same files, issues named by the same files.

## How to find it again

Build a graph (`go run ./go/ixctl/cmd/ixforge-graph -ix <ix> -lbug <lbug.exe> -into <dir>`), then:

```cypher
MATCH (f:File)<-[:CHANGED]-(c:Commit)-[:CHANGED]->(g:File) WHERE g.id = f.id RETURN count(*);
```

A count above zero means the engine is walking: add the `<>` guard.
