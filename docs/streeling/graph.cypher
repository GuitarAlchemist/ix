// Streeling University — graph queries (LadybugDB over the learnings catalog and the repo's git history).
//
// The graph is built by `go run ./cmd/ixforge-graph` (go/ixctl): Learning -IN_REPO-> Repo, Learning -TAGGED-> Tag,
// Commit -CHANGED-> File, Learning -MENTIONS-> File (a file the learning's text names). Commits that change more than
// 40 files are left out. The same questions in edge-list SQL are in graph-baseline.sql, to compare.

// 1) Related learnings: pairs that share at least two tags.
MATCH (a:Learning)-[:TAGGED]->(t:Tag)<-[:TAGGED]-(b:Learning)
WHERE a.id < b.id
WITH a, b, collect(t.name) AS tags
WHERE size(tags) >= 2
RETURN a.title AS learning, b.title AS related, tags
ORDER BY size(tags) DESC, learning
LIMIT 25;

// 2) Files that change together: the pairs most often changed by the same commit.
MATCH (a:File)<-[:CHANGED]-(c:Commit)-[:CHANGED]->(b:File)
WHERE a.path < b.path
RETURN a.path AS file, b.path AS with_file, count(c) AS together
ORDER BY together DESC, file
LIMIT 25;

// 3) Files the learnings name most: where the hard-won knowledge is.
MATCH (l:Learning)-[:MENTIONS]->(f:File)
RETURN f.path AS file, count(l) AS learnings, collect(l.title) AS titles
ORDER BY learnings DESC, file
LIMIT 20;

// 4) Read before you touch: learnings that name a file changing together (at least twice) with a file of the last 30 commits.
MATCH (r:Commit)
WITH r ORDER BY r.date DESC LIMIT 30
MATCH (r)-[:CHANGED]->(f:File)
WITH DISTINCT f
MATCH (f)<-[:CHANGED]-(c:Commit)-[:CHANGED]->(g:File)
WITH f, g, count(DISTINCT c) AS together
WHERE together >= 2
MATCH (l:Learning)-[:MENTIONS]->(g)
RETURN l.title AS learning, l.path AS path, collect(DISTINCT f.path) AS because_of, max(together) AS strength
ORDER BY strength DESC, learning
LIMIT 25;

// 5) Tags that join the repos: tags carried by learnings of both ix and ga.
MATCH (:Repo {name: 'ix'})<-[:IN_REPO]-(a:Learning)-[:TAGGED]->(t:Tag)<-[:TAGGED]-(b:Learning)-[:IN_REPO]->(:Repo {name: 'ga'})
RETURN t.name AS tag, count(DISTINCT a) AS ix_learnings, count(DISTINCT b) AS ga_learnings
ORDER BY ix_learnings + ga_learnings DESC, tag
LIMIT 20;
