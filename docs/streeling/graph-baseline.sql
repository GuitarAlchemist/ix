-- Streeling University — the questions of graph.cypher in edge-list SQL (DuckDB over the CSV files ixforge-graph
-- writes), to compare answers and times with LadybugDB. Run from the graph's output folder:
--   duckdb -c ".read <ix>/docs/streeling/graph-baseline.sql"

-- 1) Related learnings: pairs that share at least two tags.
WITH tagged AS (SELECT * FROM read_csv('tagged.csv')), learning AS (SELECT * FROM read_csv('learning.csv'))
SELECT la.title AS learning, lb.title AS related, list(a."to" ORDER BY a."to") AS tags
FROM tagged a JOIN tagged b ON a."to" = b."to" AND a."from" < b."from"
JOIN learning la ON la.id = a."from" JOIN learning lb ON lb.id = b."from"
GROUP BY la.title, lb.title
HAVING count(*) >= 2
ORDER BY count(*) DESC, learning
LIMIT 25;

-- 2) Files that change together: the pairs most often changed by the same commit.
WITH changed AS (SELECT * FROM read_csv('changed.csv'))
SELECT a."to" AS file, b."to" AS with_file, count(*) AS together
FROM changed a JOIN changed b ON a."from" = b."from" AND a."to" < b."to"
GROUP BY 1, 2
ORDER BY together DESC, file
LIMIT 25;

-- 3) Files the learnings name most: where the hard-won knowledge is.
WITH mentions AS (SELECT * FROM read_csv('mentions.csv')), learning AS (SELECT * FROM read_csv('learning.csv'))
SELECT m."to" AS file, count(*) AS learnings, list(l.title) AS titles
FROM mentions m JOIN learning l ON l.id = m."from"
GROUP BY 1
ORDER BY learnings DESC, file
LIMIT 20;

-- 4) Read before you touch: learnings that name a file changing together (at least twice) with a file of the last 30 commits.
WITH changed AS (SELECT * FROM read_csv('changed.csv')),
     commits AS (SELECT * FROM read_csv('commit.csv')),
     mentions AS (SELECT * FROM read_csv('mentions.csv')),
     learning AS (SELECT * FROM read_csv('learning.csv')),
     recent AS (SELECT DISTINCT c."to" AS f FROM changed c
                JOIN (SELECT sha FROM commits ORDER BY date DESC LIMIT 30) r ON r.sha = c."from"),
     near AS (SELECT a."to" AS f, b."to" AS g, count(DISTINCT a."from") AS together
              FROM changed a JOIN changed b ON a."from" = b."from" AND a."to" <> b."to"
              WHERE a."to" IN (SELECT f FROM recent)
              GROUP BY 1, 2 HAVING together >= 2)
SELECT l.title AS learning, l.path, list(DISTINCT n.f) AS because_of, max(n.together) AS strength
FROM near n JOIN mentions m ON m."to" = n.g JOIN learning l ON l.id = m."from"
GROUP BY l.title, l.path
ORDER BY strength DESC, learning
LIMIT 25;

-- 5) Tags that join the repos: tags carried by learnings of both ix and ga.
WITH tagged AS (SELECT * FROM read_csv('tagged.csv')), in_repo AS (SELECT * FROM read_csv('in_repo.csv'))
SELECT t."to" AS tag,
       count(DISTINCT t."from") FILTER (WHERE r."to" = 'ix') AS ix_learnings,
       count(DISTINCT t."from") FILTER (WHERE r."to" = 'ga') AS ga_learnings
FROM tagged t JOIN in_repo r ON r."from" = t."from"
GROUP BY 1
HAVING ix_learnings > 0 AND ga_learnings > 0
ORDER BY ix_learnings + ga_learnings DESC, tag
LIMIT 20;
