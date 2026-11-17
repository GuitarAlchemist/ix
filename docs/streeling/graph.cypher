// Streeling University — graph queries (LadybugDB over the learnings catalog, the git history of ix and ga, and their
// pull requests and issues).
//
// The graph is built by `go run ./cmd/ixforge-graph` (go/ixctl): Learning -IN_REPO-> Repo, Learning -TAGGED-> Tag,
// Commit -CHANGED-> File, Learning -MENTIONS-> File (a file the learning's text names), PR -TOUCHES-> File,
// PR -CLOSES-> Issue, Issue -NAMES-> File. A file is known by its repo and path (File.id is repo:path). Commits that
// change more than 40 files are left out. The Desktop runs these queries on its newest build; queries 1 to 5 in
// edge-list SQL are in graph-baseline.sql, to compare.

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
WHERE a.id < b.id
RETURN a.repo AS repo, a.path AS file, b.path AS with_file, count(c) AS together
ORDER BY together DESC, repo, file
LIMIT 25;

// 3) Files the learnings name most: where the hard-won knowledge is.
MATCH (l:Learning)-[:MENTIONS]->(f:File)
RETURN f.repo AS repo, f.path AS file, count(l) AS learnings, collect(l.title) AS titles
ORDER BY learnings DESC, repo, file
LIMIT 20;

// 4) Read before you touch: learnings that name a file changing together (at least twice) with a file of the last 30 commits.
MATCH (r:Commit)
WITH r ORDER BY r.date DESC, r.sha LIMIT 30
MATCH (r)-[:CHANGED]->(f:File)
WITH DISTINCT f
MATCH (f)<-[:CHANGED]-(c:Commit)-[:CHANGED]->(g:File)
WHERE g.id <> f.id
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

// 6) Open pull requests that collide: pairs that touch the same files.
MATCH (a:PR {state: 'OPEN'})-[:TOUCHES]->(f:File)<-[:TOUCHES]-(b:PR {state: 'OPEN'})
WHERE a.id < b.id
RETURN a.id AS pr, a.title AS title, b.id AS with_pr, b.title AS with_title, count(f) AS shared, collect(f.path) AS files
ORDER BY shared DESC, pr
LIMIT 25;

// 7) What an open pull request should read first: the learnings that name a file it touches.
MATCH (p:PR {state: 'OPEN'})-[:TOUCHES]->(f:File)<-[:MENTIONS]-(l:Learning)
RETURN p.id AS pr, p.title AS title, l.title AS learning, l.path AS path, collect(f.path) AS files
ORDER BY pr DESC, learning
LIMIT 40;

// 8) Open issues and the open pull requests that touch the files they name, without closing them.
MATCH (i:Issue {state: 'OPEN'})-[:NAMES]->(f:File)<-[:TOUCHES]-(p:PR {state: 'OPEN'})
WHERE NOT (p)-[:CLOSES]->(i)
RETURN i.id AS issue, i.title AS title, p.id AS pr, p.title AS pr_title, collect(f.path) AS files
ORDER BY issue DESC, pr
LIMIT 25;

// 9) Files fixed again and again: the files closed issues name, by how many issues, with the learnings that name them.
MATCH (p:PR {state: 'MERGED'})-[:CLOSES]->(i:Issue)-[:NAMES]->(f:File)
WITH f, count(DISTINCT i) AS issues
OPTIONAL MATCH (l:Learning)-[:MENTIONS]->(f)
RETURN f.repo AS repo, f.path AS file, issues, collect(DISTINCT l.title) AS learnings
ORDER BY issues DESC, repo, file
LIMIT 20;
