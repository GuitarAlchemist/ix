-- Streeling University — questions of graph.cypher in edge-list SQL (DuckDB over the CSV files of an ixforge-graph
-- build), to compare answers and times with LadybugDB. Run from a build folder:
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
WITH changed AS (SELECT * FROM read_csv('changed.csv')), file AS (SELECT * FROM read_csv('file.csv'))
SELECT fa.repo, fa.path AS file, fb.path AS with_file, count(*) AS together
FROM changed a JOIN changed b ON a."from" = b."from" AND a."to" < b."to"
JOIN file fa ON fa.id = a."to" JOIN file fb ON fb.id = b."to"
GROUP BY 1, 2, 3
ORDER BY together DESC, repo, file
LIMIT 25;

-- 3) Files the learnings name most: where the hard-won knowledge is.
WITH mentions AS (SELECT * FROM read_csv('mentions.csv')), learning AS (SELECT * FROM read_csv('learning.csv')),
     file AS (SELECT * FROM read_csv('file.csv'))
SELECT f.repo, f.path AS file, count(*) AS learnings, list(l.title) AS titles
FROM mentions m JOIN learning l ON l.id = m."from" JOIN file f ON f.id = m."to"
GROUP BY 1, 2
ORDER BY learnings DESC, repo, file
LIMIT 20;

-- 4) Read before you touch: learnings that name a file changing together (at least twice) with a file of the last 30 commits.
WITH changed AS (SELECT * FROM read_csv('changed.csv')),
     commits AS (SELECT * FROM read_csv('commit.csv')),
     mentions AS (SELECT * FROM read_csv('mentions.csv')),
     learning AS (SELECT * FROM read_csv('learning.csv')),
     file AS (SELECT * FROM read_csv('file.csv')),
     recent AS (SELECT DISTINCT c."to" AS f FROM changed c
                JOIN (SELECT sha FROM commits ORDER BY date DESC, sha LIMIT 30) r ON r.sha = c."from"),
     near AS (SELECT a."to" AS f, b."to" AS g, count(DISTINCT a."from") AS together
              FROM changed a JOIN changed b ON a."from" = b."from" AND a."to" <> b."to"
              WHERE a."to" IN (SELECT f FROM recent)
              GROUP BY 1, 2 HAVING together >= 2)
SELECT l.title AS learning, l.path, list(DISTINCT fp.path) AS because_of, max(n.together) AS strength
FROM near n JOIN mentions m ON m."to" = n.g JOIN learning l ON l.id = m."from" JOIN file fp ON fp.id = n.f
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

-- 6) Open pull requests that collide: pairs that touch the same files.
WITH touches AS (SELECT * FROM read_csv('touches.csv')), pr AS (SELECT * FROM read_csv('pr.csv')),
     file AS (SELECT * FROM read_csv('file.csv'))
SELECT a.id AS pr, a.title, b.id AS with_pr, b.title AS with_title, count(*) AS shared, list(f.path) AS files
FROM touches ta JOIN touches tb ON ta."to" = tb."to" AND ta."from" < tb."from"
JOIN pr a ON a.id = ta."from" AND a.state = 'OPEN' JOIN pr b ON b.id = tb."from" AND b.state = 'OPEN'
JOIN file f ON f.id = ta."to"
GROUP BY a.id, a.title, b.id, b.title
ORDER BY shared DESC, pr
LIMIT 25;

-- 8) Open issues and the open pull requests that touch the files they name, without closing them.
WITH names AS (SELECT * FROM read_csv('names.csv')), touches AS (SELECT * FROM read_csv('touches.csv')),
     closes AS (SELECT * FROM read_csv('closes.csv')), pr AS (SELECT * FROM read_csv('pr.csv')),
     issue AS (SELECT * FROM read_csv('issue.csv')), file AS (SELECT * FROM read_csv('file.csv'))
SELECT i.id AS issue, i.title, p.id AS pr, p.title AS pr_title, list(f.path) AS files
FROM names n JOIN issue i ON i.id = n."from" AND i.state = 'OPEN'
JOIN touches t ON t."to" = n."to" JOIN pr p ON p.id = t."from" AND p.state = 'OPEN'
JOIN file f ON f.id = n."to"
WHERE NOT EXISTS (SELECT 1 FROM closes c WHERE c."from" = p.id AND c."to" = i.id)
GROUP BY i.id, i.title, p.id, p.title
ORDER BY issue DESC, pr
LIMIT 25;
