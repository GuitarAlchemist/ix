package workshop

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os/exec"
	"strings"
	"sync"
	"time"

	"github.com/GuitarAlchemist/ix/go/ixctl/internal/repograph"
)

// The Desktop runs the listed DuckDB queries (owner's choice, 2026-11-16): any approved device names one query of the
// queries widget by its file and title, and the Desktop reads that query's SQL from the ix repo itself, runs it with the
// duckdb CLI in the repo's folder (the queries read repo files such as state/streeling/catalog.jsonl), and answers its
// rows. A device never sends SQL: what runs is what the repo's files say. One query at a time, stopped after a time
// limit; a result too large to send is refused whole, never cut.
//
// The queries of a .cypher file (the Streeling graph, owner's choice, 2026-11-16) run the same way with the LadybugDB
// CLI, read-only, on the newest finished build of the repo's graph (state/graph, written by cmd/ixforge-graph).

const (
	maxQueryOutput      = 1 << 20
	maxQueryError       = 4 << 10
	defaultQueryTimeout = 60 * time.Second
)

var errQueryUnknown = errors.New("query_unknown")

// QueryRunner runs the queries of Files, relative to Repo, with the duckdb executable DuckDB, and those of .cypher files
// with the LadybugDB executable Lbug on the newest finished graph build of Graphs.
type QueryRunner struct {
	Repo    string
	Files   []string
	DuckDB  string
	Lbug    string
	Graphs  string
	Timeout time.Duration
	// Run runs sql in dir and returns DuckDB's standard output and error (tests replace it); nil runs DuckDB.
	Run func(ctx context.Context, dir, sql string) (stdout, stderr []byte, err error)
	// RunGraph runs a Cypher query on db, read-only, and returns LadybugDB's output (tests replace it); nil runs Lbug.
	RunGraph func(ctx context.Context, db, query string) ([]byte, error)

	mu sync.Mutex
}

// WithQueries lets every approved device run the listed queries on the Desktop.
func WithQueries(r *QueryRunner) Option { return func(o *handlerOptions) { o.queries = r } }

// QueryResult is one run: the SQL the Desktop read and ran, and either its rows (a JSON array, as DuckDB wrote it) or
// an error (DuckDB's own message, query_timed_out or query_result_too_large).
type QueryResult struct {
	Source string          `json:"source"`
	Title  string          `json:"title"`
	SQL    string          `json:"sql"`
	Rows   json.RawMessage `json:"rows,omitempty"`
	Error  string          `json:"error,omitempty"`
	MS     int64           `json:"ms"`
}

// find reads the listed query named by source and title from the repo.
func (r *QueryRunner) find(source, title string) (QueryEntry, error) {
	listed := false
	for _, f := range r.Files {
		listed = listed || f == source
	}
	if !listed {
		return QueryEntry{}, errQueryUnknown
	}
	entries, err := ReadQueries(r.Repo, []string{source})
	if err != nil {
		return QueryEntry{}, err
	}
	for _, e := range entries {
		if e.Title == title {
			return e, nil
		}
	}
	return QueryEntry{}, errQueryUnknown
}

// run runs sql with the duckdb CLI: JSON output, the SQL on standard input, stopping at the first error.
func (r *QueryRunner) run(ctx context.Context, dir, sql string) ([]byte, []byte, error) {
	if r.Run != nil {
		return r.Run(ctx, dir, sql)
	}
	cmd := exec.CommandContext(ctx, r.DuckDB, "-json", "-bail")
	cmd.Dir = dir
	cmd.Stdin = strings.NewReader(sql)
	out := &cappedBuffer{max: maxQueryOutput + 1}
	errOut := &cappedBuffer{max: maxQueryError}
	cmd.Stdout, cmd.Stderr = out, errOut
	cmd.WaitDelay = 2 * time.Second
	err := cmd.Run()
	return out.buf.Bytes(), errOut.buf.Bytes(), err
}

// runGraph runs a Cypher query with the lbug CLI on db: read-only, JSON output, the query on standard input. lbug
// prints its errors on its output.
func (r *QueryRunner) runGraph(ctx context.Context, db, query string) ([]byte, error) {
	if r.RunGraph != nil {
		return r.RunGraph(ctx, db, query)
	}
	cmd := exec.CommandContext(ctx, r.Lbug, db, "-r", "-b", "-s", "-m", "json")
	cmd.Stdin = strings.NewReader(query)
	out := &cappedBuffer{max: maxQueryOutput + 1}
	cmd.Stdout, cmd.Stderr = out, out
	cmd.WaitDelay = 2 * time.Second
	err := cmd.Run()
	return out.buf.Bytes(), err
}

// lastArray is the last JSON value of DuckDB's output (a statement that returns no rows writes nothing; one that sets
// a variable before the query writes nothing either), or an empty array.
func lastArray(out []byte) (json.RawMessage, bool) {
	d := json.NewDecoder(bytes.NewReader(out))
	last := json.RawMessage("[]")
	for {
		var v json.RawMessage
		if err := d.Decode(&v); err == io.EOF {
			return last, true
		} else if err != nil || !bytes.HasPrefix(bytes.TrimSpace(v), []byte("[")) {
			return nil, false
		}
		last = v
	}
}

// Query runs one listed query.
func (r *QueryRunner) Query(ctx context.Context, source, title string) (QueryResult, error) {
	entry, err := r.find(source, title)
	if err != nil {
		return QueryResult{}, err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	limit := r.Timeout
	if limit <= 0 {
		limit = defaultQueryTimeout
	}
	runCtx, cancel := context.WithTimeout(ctx, limit)
	defer cancel()
	result := QueryResult{Source: entry.Source, Title: entry.Title, SQL: entry.SQL}
	started := time.Now()
	var out, errOut []byte
	var runErr error
	if strings.HasSuffix(entry.Source, ".cypher") {
		if r.Lbug == "" {
			result.Error = "graph_engine_missing"
			return result, nil
		}
		db, err := repograph.Latest(r.Graphs)
		if errors.Is(err, repograph.ErrNoGraph) {
			result.Error = "graph_missing"
			return result, nil
		} else if err != nil {
			return QueryResult{}, err
		}
		out, runErr = r.runGraph(runCtx, db, entry.SQL)
		if line := repograph.ErrorLine(out); line != "" && runErr == nil {
			errOut, runErr = []byte(line), errors.New("lbug reported an error")
		} else if runErr != nil && len(out) <= maxQueryError {
			errOut = out // what lbug said before it failed
		}
	} else {
		out, errOut, runErr = r.run(runCtx, r.Repo, entry.SQL)
	}
	result.MS = time.Since(started).Milliseconds()
	switch {
	case runCtx.Err() != nil:
		result.Error = "query_timed_out"
	case runErr != nil:
		result.Error = strings.TrimSpace(strings.ToValidUTF8(string(errOut), "�"))
		if result.Error == "" {
			result.Error = runErr.Error()
		}
	case len(out) > maxQueryOutput:
		result.Error = "query_result_too_large"
	default:
		rows, ok := lastArray(out)
		if !ok {
			result.Error = "query_output_unreadable"
			break
		}
		result.Rows = rows
	}
	return result, nil
}

func queryRoutes(mux *http.ServeMux, s *Store, r *QueryRunner) {
	mux.HandleFunc("POST /workshop/api/queries/run", func(w http.ResponseWriter, req *http.Request) {
		if _, err := s.Me(bearer(req)); err != nil {
			failure(w, err)
			return
		}
		var body struct {
			Source string `json:"source"`
			Title  string `json:"title"`
		}
		if err := decode(w, req, &body); err != nil {
			failure(w, ErrInvalid)
			return
		}
		result, err := r.Query(req.Context(), body.Source, body.Title)
		switch {
		case errors.Is(err, errQueryUnknown):
			failureStatus(w, http.StatusBadRequest, err.Error())
		case err != nil:
			failure(w, err)
		default:
			respond(w, http.StatusOK, result)
		}
	})
}
