//! TEMPORARY exploratory probe — deleted before commit.

use ix_grammar::ebnf;

fn probe(label: &str, src: &str) {
    println!("\n════ {label}");
    println!("  input: {src:?}");
    match ebnf::parse(src) {
        Ok(g) => println!("  ISO   OK   start={} rules={}", g.start, g.productions.len()),
        Err(e) => println!("  ISO   ERR  line={} col={} :: {}", e.line, e.col, e.message),
    }
}

#[test]
fn probe4_positions_after_fix() {
    // Error is on line 4, col 5 in every one of these. Only the amount
    // of preceding comment text differs.
    probe("no-comment", "\n\n\nA = ) ;\n");
    probe("comment-3-lines", "(* line1\nline2\nline3 *)\nA = ) ;\n");
    probe("comment-2-lines", "(* a\nb *)\n\nA = ) ;\n");

    // Column: `)` sits at col 19 in both.
    probe("col-inline-comment", "A = (* xxxxxxx *) ) ;\n");
    probe("col-no-comment", "A =               ) ;\n");

    // Productions must be unaffected by comments/whitespace.
    for (l, s) in [
        ("plain", "A = \"x\" ;\nB = A ;\n"),
        (
            "commented",
            "(* h *)\nA (* a *) = (* b *) \"x\" (* c *) ;\n(* d *)\nB = A ;\n",
        ),
        ("spaced", "\n\nA   =    \"x\"   ;\n\n\nB\t=\tA\t;\n"),
    ] {
        let g = ebnf::parse(s).expect("parse");
        let mut keys: Vec<_> = g.productions.keys().cloned().collect();
        keys.sort();
        println!("  {l}: start={} keys={:?}", g.start, keys);
    }
}
