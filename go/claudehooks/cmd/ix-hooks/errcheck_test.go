package main

import (
	"go/ast"
	"go/types"
	"testing"
)

// Every error a hook gets is handled, or discarded on purpose as `_ = f()`, so the reviewer sees
// each discard in the diff. A silently dropped error lost the previous digest once: a failed copy
// to state/digests/archive, then latest.md overwritten (PR #512 review). Printing to stdout or
// stderr (fmt.Print*, fmt.Fprint*) is exempt, as in errcheck.
func TestNoErrorIsDroppedSilently(t *testing.T) {
	_, files, info := checkedPackage(t)
	errType := types.Universe.Lookup("error").Type()
	returnsError := func(tv types.TypeAndValue) bool {
		if tuple, ok := tv.Type.(*types.Tuple); ok {
			for i := 0; i < tuple.Len(); i++ {
				if types.Identical(tuple.At(i).Type(), errType) {
					return true
				}
			}
			return false
		}
		return tv.Type != nil && types.Identical(tv.Type, errType)
	}
	for _, f := range files {
		ast.Inspect(f, func(n ast.Node) bool {
			stmt, ok := n.(*ast.ExprStmt)
			if !ok {
				return true
			}
			call, ok := stmt.X.(*ast.CallExpr)
			if !ok || !returnsError(info.Types[call]) {
				return true
			}
			if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
				if fn, ok := info.Uses[sel.Sel].(*types.Func); ok && fn.Pkg() != nil && fn.Pkg().Path() == "fmt" {
					return true
				}
				if recv := info.Types[sel.X].Type; recv != nil {
					switch recv.String() {
					case "*strings.Builder", "strings.Builder", "*bytes.Buffer", "bytes.Buffer":
						return true // their writes never fail
					}
				}
			}
			t.Errorf("%s: the error of %s is dropped; handle it or write `_ = …`", checkedFset.Position(call.Pos()), types.ExprString(call.Fun))
			return true
		})
	}
}
