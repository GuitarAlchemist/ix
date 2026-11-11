package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// write creates dir/rel with body, making parent directories as needed.
func write(t *testing.T, dir, rel, body string) {
	t.Helper()
	p := filepath.Join(dir, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func hashOf(t *testing.T, dir string) string {
	t.Helper()
	h, err := sourceHash(dir)
	if err != nil {
		t.Fatalf("sourceHash(%s): %v", dir, err)
	}
	return h
}

func TestSourceHashUnchangedIsStable(t *testing.T) {
	dir := t.TempDir()
	write(t, dir, "main.go", "package main\n")
	a := hashOf(t, dir)
	if b := hashOf(t, dir); a != b {
		t.Fatalf("same content hashed twice: %s != %s", a, b)
	}
	// Rewriting identical bytes (new mtime) must not change the hash.
	write(t, dir, "main.go", "package main\n")
	if b := hashOf(t, dir); a != b {
		t.Fatalf("rewrite of identical content changed hash: %s != %s", a, b)
	}
	if len(a) != 12 {
		t.Fatalf("hash length = %d, want 12", len(a))
	}
}

func TestSourceHashContentChange(t *testing.T) {
	dir := t.TempDir()
	write(t, dir, "main.go", "package main\n")
	a := hashOf(t, dir)
	write(t, dir, "main.go", "package main\n// edit\n")
	if b := hashOf(t, dir); a == b {
		t.Fatal("edited content kept the same hash")
	}
}

func TestSourceHashAddedFile(t *testing.T) {
	dir := t.TempDir()
	write(t, dir, "main.go", "package main\n")
	a := hashOf(t, dir)
	write(t, dir, "extra.go", "package main\n")
	if b := hashOf(t, dir); a == b {
		t.Fatal("adding a .go file kept the same hash")
	}
}

func TestSourceHashRemovedFile(t *testing.T) {
	dir := t.TempDir()
	write(t, dir, "main.go", "package main\n")
	write(t, dir, "extra.go", "package main\n")
	a := hashOf(t, dir)
	if err := os.Remove(filepath.Join(dir, "extra.go")); err != nil {
		t.Fatal(err)
	}
	if b := hashOf(t, dir); a == b {
		t.Fatal("removing a .go file kept the same hash")
	}
}

func TestSourceHashRenamedFile(t *testing.T) {
	dir := t.TempDir()
	write(t, dir, "main.go", "package main\n")
	a := hashOf(t, dir)
	if err := os.Rename(filepath.Join(dir, "main.go"), filepath.Join(dir, "probe.go")); err != nil {
		t.Fatal(err)
	}
	if b := hashOf(t, dir); a == b {
		t.Fatal("renaming a file with identical content kept the same hash")
	}
}

func TestSourceHashSwappedContentsBetweenFiles(t *testing.T) {
	// Names are part of the hash: the same two bodies under swapped names differ.
	d1, d2 := t.TempDir(), t.TempDir()
	write(t, d1, "a.go", "package a\n")
	write(t, d1, "b.go", "package b\n")
	write(t, d2, "a.go", "package b\n")
	write(t, d2, "b.go", "package a\n")
	if hashOf(t, d1) == hashOf(t, d2) {
		t.Fatal("swapping contents between names kept the same hash")
	}
}

func TestSourceHashBoundaryAmbiguity(t *testing.T) {
	// Moving bytes across the file boundary must not collide.
	d1, d2 := t.TempDir(), t.TempDir()
	write(t, d1, "a.go", "xy")
	write(t, d1, "b.go", "z")
	write(t, d2, "a.go", "x")
	write(t, d2, "b.go", "yz")
	if hashOf(t, d1) == hashOf(t, d2) {
		t.Fatal("different file splits of the same bytes collided")
	}
}

func TestSourceHashIndependentOfDirLocationAndCreationOrder(t *testing.T) {
	d1, d2 := t.TempDir(), t.TempDir()
	write(t, d1, "a.go", "package a\n")
	write(t, d1, "sub/b.go", "package b\n")
	write(t, d2, "sub/b.go", "package b\n")
	write(t, d2, "a.go", "package a\n")
	if hashOf(t, d1) != hashOf(t, d2) {
		t.Fatal("same tree in a different directory or creation order hashed differently")
	}
}

func TestSourceHashNestedPaths(t *testing.T) {
	dir := t.TempDir()
	write(t, dir, "main.go", "package main\n")
	write(t, dir, "internal/deep/util.go", "package deep\n")
	a := hashOf(t, dir)

	write(t, dir, "internal/deep/util.go", "package deep\n// edit\n")
	b := hashOf(t, dir)
	if a == b {
		t.Fatal("edit to a nested file kept the same hash")
	}

	// Same nested content at a different depth is a different tree.
	if err := os.Rename(filepath.Join(dir, "internal/deep/util.go"), filepath.Join(dir, "internal/util.go")); err != nil {
		t.Fatal(err)
	}
	if c := hashOf(t, dir); c == b {
		t.Fatal("moving a nested file to another directory kept the same hash")
	}
}

func TestSourceHashIgnoresNonGoFiles(t *testing.T) {
	dir := t.TempDir()
	write(t, dir, "main.go", "package main\n")
	a := hashOf(t, dir)

	write(t, dir, "README.md", "notes\n")
	write(t, dir, "data.json", "{}\n")
	write(t, dir, "sub/notes.txt", "x\n")
	write(t, dir, "main.go.bak", "package old\n")
	write(t, dir, "gofile", "package x\n")
	write(t, dir, "main.go.tmpl", "package {{.}}\n")
	if b := hashOf(t, dir); a != b {
		t.Fatalf("non-.go files changed the hash: %s != %s", a, b)
	}

	write(t, dir, "README.md", "changed notes\n")
	if b := hashOf(t, dir); a != b {
		t.Fatal("editing a non-.go file changed the hash")
	}
}

func TestSourceHashDirectoryNamedDotGoIsNotAFile(t *testing.T) {
	dir := t.TempDir()
	write(t, dir, "main.go", "package main\n")
	a := hashOf(t, dir)
	if err := os.MkdirAll(filepath.Join(dir, "vendor.go"), 0o755); err != nil {
		t.Fatal(err)
	}
	if b := hashOf(t, dir); a != b {
		t.Fatal("an empty directory named *.go changed the hash")
	}
}

func TestSourceHashEmptyDirectory(t *testing.T) {
	dir := t.TempDir()
	h, err := sourceHash(dir)
	if err == nil {
		t.Fatalf("empty dir: want error, got hash %q", h)
	}
	if !strings.Contains(err.Error(), "no .go files") {
		t.Fatalf("empty dir error = %v, want it to mention no .go files", err)
	}
}

func TestSourceHashOnlyNonGoFilesIsAnError(t *testing.T) {
	dir := t.TempDir()
	write(t, dir, "README.md", "x\n")
	if _, err := sourceHash(dir); err == nil {
		t.Fatal("dir with no .go files: want error")
	}
}

func TestSourceHashMissingDirectory(t *testing.T) {
	if _, err := sourceHash(filepath.Join(t.TempDir(), "absent")); err == nil {
		t.Fatal("missing dir: want error")
	}
}
