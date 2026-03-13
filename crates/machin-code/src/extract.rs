use ndarray::{Array1, Array2};
use crate::error::CodeError;
use crate::parse::CodeTree;

/// Maximum named nodes for dense adjacency matrix (1K nodes = 8MB).
const MAX_DENSE_NAMED_NODES: usize = 1_000;

/// Walk all nodes in pre-order using a TreeCursor (iterative, zero-allocation traversal).
///
/// Calls `visit` for every node in the tree. This is the single traversal
/// primitive — all extraction functions build on it.
fn walk_preorder(tree: &CodeTree, mut visit: impl FnMut(tree_sitter::Node)) {
    let mut cursor = tree.tree().walk();
    loop {
        visit(cursor.node());
        if cursor.goto_first_child() {
            continue;
        }
        loop {
            if cursor.goto_next_sibling() {
                break;
            }
            if !cursor.goto_parent() {
                return;
            }
        }
    }
}

/// Extract a node-kind histogram from a parsed code tree.
///
/// Uses `kind_id()` array indexing for O(N) performance with excellent
/// cache locality (~1.2KB for a typical grammar vocabulary of 200-300 kinds).
///
/// Both named and anonymous nodes are counted. ERROR and MISSING nodes
/// from partial parses get their own histogram entries.
pub fn histogram(tree: &CodeTree) -> Array1<f64> {
    let vocab = tree.language().node_kind_count();
    let mut counts = vec![0u32; vocab];
    walk_preorder(tree, |node| {
        let id = node.kind_id() as usize;
        if id < counts.len() {
            counts[id] += 1;
        }
    });
    Array1::from_vec(counts.into_iter().map(|c| c as f64).collect())
}

/// Extract a histogram counting only named nodes (grammar symbols, not punctuation).
///
/// This is typically more useful for ML since it ignores syntax tokens like
/// `{`, `}`, `;` and focuses on semantic structure.
pub fn histogram_named(tree: &CodeTree) -> Array1<f64> {
    let vocab = tree.language().node_kind_count();
    let mut counts = vec![0u32; vocab];
    walk_preorder(tree, |node| {
        if node.is_named() {
            let id = node.kind_id() as usize;
            if id < counts.len() {
                counts[id] += 1;
            }
        }
    });
    Array1::from_vec(counts.into_iter().map(|c| c as f64).collect())
}

/// Get the vocabulary mapping: histogram index -> node kind name.
///
/// The returned Vec has the same length as the histogram. Index `i` corresponds
/// to the node kind with `kind_id == i`.
pub fn histogram_vocabulary(tree: &CodeTree) -> Vec<String> {
    let vocab = tree.language().node_kind_count();
    assert!(
        vocab <= u16::MAX as usize + 1,
        "grammar vocabulary ({vocab}) exceeds u16 range"
    );
    (0..vocab)
        .map(|id| {
            tree.language()
                .node_kind_for_id(id as u16)
                .unwrap_or("unknown")
                .to_string()
        })
        .collect()
}

/// Count total named nodes in the tree.
pub fn named_node_count(tree: &CodeTree) -> usize {
    let mut count = 0usize;
    walk_preorder(tree, |node| {
        if node.is_named() {
            count += 1;
        }
    });
    count
}

/// Extract a dense adjacency matrix for named nodes only.
///
/// Rows and columns correspond to named nodes in pre-order traversal.
/// A non-zero entry `matrix[[i, j]] = 1.0` means node `i` is the parent of node `j`.
///
/// # Errors
/// Returns `CodeError::TreeTooLarge` if the tree has more than 1,000 named nodes.
pub fn adjacency(tree: &CodeTree) -> Result<Array2<f64>, CodeError> {
    // Single pass: collect named node IDs (merges old pass 1 + pass 2)
    let mut node_ids: Vec<usize> = Vec::new();
    walk_preorder(tree, |node| {
        if node.is_named() {
            node_ids.push(node.id());
        }
    });

    let n = node_ids.len();
    if n > MAX_DENSE_NAMED_NODES {
        return Err(CodeError::TreeTooLarge(n, MAX_DENSE_NAMED_NODES));
    }
    if n == 0 {
        return Ok(Array2::zeros((0, 0)));
    }

    // Build id-to-index lookup
    let id_to_index: std::collections::HashMap<usize, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(idx, &id)| (id, idx))
        .collect();

    // Fill adjacency matrix (iterative — no recursion, no stack overflow)
    let mut matrix = Array2::<f64>::zeros((n, n));
    let mut cursor = tree.tree().walk();
    let mut parent_stack: Vec<Option<usize>> = Vec::new();

    loop {
        let node = cursor.node();
        let node_idx = if node.is_named() {
            id_to_index.get(&node.id()).copied()
        } else {
            None
        };

        // Add edge from nearest named ancestor to this named node
        if let Some(ci) = node_idx {
            if let Some(&Some(pi)) = parent_stack.last() {
                matrix[[pi, ci]] = 1.0;
            }
        }

        if cursor.goto_first_child() {
            // Push the nearest named ancestor index for children
            let named_ancestor = node_idx.or_else(|| {
                parent_stack.last().copied().flatten()
            });
            parent_stack.push(named_ancestor);
            continue;
        }
        loop {
            if cursor.goto_next_sibling() {
                break;
            }
            parent_stack.pop();
            if !cursor.goto_parent() {
                return Ok(matrix);
            }
        }
    }
}

/// Get node kind labels for the adjacency matrix rows/columns.
///
/// Returns the kind name for each named node in pre-order traversal order,
/// matching the row/column ordering of `adjacency()`.
pub fn adjacency_labels(tree: &CodeTree) -> Vec<String> {
    let mut labels = Vec::new();
    walk_preorder(tree, |node| {
        if node.is_named() {
            labels.push(node.kind().to_string());
        }
    });
    labels
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    fn parse_rust(source: &str) -> CodeTree {
        parse::parse("rust", source).unwrap()
    }

    #[test]
    fn test_histogram_basic() {
        let tree = parse_rust("fn main() { let x = 42; }");
        let hist = histogram(&tree);

        // Histogram length == vocabulary size
        assert_eq!(hist.len(), tree.language().node_kind_count());
        // Total count should be > 0
        assert!(hist.sum() > 0.0);
    }

    #[test]
    fn test_histogram_named_basic() {
        let tree = parse_rust("fn main() { let x = 42; }");
        let hist_all = histogram(&tree);
        let hist_named = histogram_named(&tree);

        // Named histogram should have fewer total counts (no punctuation)
        assert!(hist_named.sum() < hist_all.sum());
        assert!(hist_named.sum() > 0.0);
    }

    #[test]
    fn test_histogram_vocabulary() {
        let tree = parse_rust("fn main() {}");
        let vocab = histogram_vocabulary(&tree);
        let hist = histogram(&tree);

        assert_eq!(vocab.len(), hist.len());
        // Should contain known Rust node kinds
        assert!(vocab.contains(&"function_item".to_string()));
        assert!(vocab.contains(&"identifier".to_string()));
    }

    #[test]
    fn test_histogram_specific_kinds() {
        let tree = parse_rust("fn main() { let x = 42; }");
        let hist = histogram_named(&tree);
        let vocab = histogram_vocabulary(&tree);

        // Find indices for known kinds
        let func_idx = vocab.iter().position(|k| k == "function_item").unwrap();
        let let_idx = vocab.iter().position(|k| k == "let_declaration").unwrap();
        let int_idx = vocab.iter().position(|k| k == "integer_literal").unwrap();

        assert!(hist[func_idx] > 0.0, "function_item should be present");
        assert!(hist[let_idx] > 0.0, "let_declaration should be present");
        assert!(hist[int_idx] > 0.0, "integer_literal should be present");
    }

    #[test]
    fn test_histogram_with_errors() {
        let tree = parse_rust("fn foo( {");
        assert!(tree.has_errors());
        let hist = histogram(&tree);
        // Should still produce a valid histogram
        assert!(hist.sum() > 0.0);
    }

    #[test]
    fn test_named_node_count() {
        let tree = parse_rust("fn main() { let x = 42; }");
        let count = named_node_count(&tree);
        assert!(count > 0);
        assert!(count < 50); // Small snippet — sanity check
    }

    #[test]
    fn test_adjacency_basic() {
        let tree = parse_rust("fn main() { let x = 42; }");
        let matrix = adjacency(&tree).unwrap();
        let n = named_node_count(&tree);

        assert_eq!(matrix.shape(), &[n, n]);
        // Should have some edges (parent-child relationships)
        assert!(matrix.sum() > 0.0);
        // Diagonal should be zero (no self-loops)
        for i in 0..n {
            assert!((matrix[[i, i]]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_adjacency_labels() {
        let tree = parse_rust("fn main() {}");
        let labels = adjacency_labels(&tree);
        let n = named_node_count(&tree);
        assert_eq!(labels.len(), n);
        // First label should be source_file (root)
        assert_eq!(labels[0], "source_file");
    }

    #[test]
    fn test_adjacency_empty_body() {
        let tree = parse_rust("fn main() {}");
        let matrix = adjacency(&tree).unwrap();
        assert!(matrix.shape()[0] > 0);
    }

    #[test]
    fn test_histogram_different_snippets_differ() {
        let tree_fn = parse_rust("fn main() { let x = 42; let y = 10; }");
        let tree_struct = parse_rust("struct Point { x: f64, y: f64 }");

        let hist_fn = histogram_named(&tree_fn);
        let hist_struct = histogram_named(&tree_struct);

        // Different code should produce different histograms
        let diff = (&hist_fn - &hist_struct).mapv(|x| x.abs()).sum();
        assert!(diff > 0.0, "different snippets should have different histograms");
    }

    #[test]
    fn test_adjacency_parent_child_edges() {
        // Verify specific parent-child edges in the adjacency matrix
        let tree = parse_rust("fn main() {}");
        let matrix = adjacency(&tree).unwrap();
        let labels = adjacency_labels(&tree);

        // source_file (idx 0) should be parent of function_item
        let func_idx = labels.iter().position(|l| l == "function_item").unwrap();
        assert!((matrix[[0, func_idx]] - 1.0).abs() < 1e-10,
            "source_file should be parent of function_item");
    }
}
