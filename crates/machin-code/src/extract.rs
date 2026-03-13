use ndarray::{Array1, Array2};
use crate::error::CodeError;
use crate::parse::CodeTree;

/// Maximum named nodes for dense adjacency matrix (1K nodes = 8MB).
const MAX_DENSE_NAMED_NODES: usize = 1_000;

/// Extract a node-kind histogram from a parsed code tree.
///
/// Uses `kind_id()` array indexing for O(N) performance with excellent
/// cache locality (~1.2KB for a typical grammar vocabulary of 200-300 kinds).
///
/// The histogram has one entry per node kind in the grammar. Use
/// `histogram_vocabulary()` to get the mapping from index to kind name.
///
/// Both named and anonymous nodes are counted. ERROR and MISSING nodes
/// from partial parses get their own histogram entries.
pub fn histogram(tree: &CodeTree) -> Array1<f64> {
    let vocab = tree.language().node_kind_count();
    let mut counts = vec![0u32; vocab];
    let mut cursor = tree.tree().walk();

    loop {
        let node = cursor.node();
        let id = node.kind_id() as usize;
        if id < counts.len() {
            counts[id] += 1;
        }
        if cursor.goto_first_child() {
            continue;
        }
        loop {
            if cursor.goto_next_sibling() {
                break;
            }
            if !cursor.goto_parent() {
                return Array1::from_vec(counts.into_iter().map(|c| c as f64).collect());
            }
        }
    }
}

/// Extract a histogram counting only named nodes (grammar symbols, not punctuation).
///
/// This is typically more useful for ML since it ignores syntax tokens like
/// `{`, `}`, `;` and focuses on semantic structure.
pub fn histogram_named(tree: &CodeTree) -> Array1<f64> {
    let vocab = tree.language().node_kind_count();
    let mut counts = vec![0u32; vocab];
    let mut cursor = tree.tree().walk();

    loop {
        let node = cursor.node();
        if node.is_named() {
            let id = node.kind_id() as usize;
            if id < counts.len() {
                counts[id] += 1;
            }
        }
        if cursor.goto_first_child() {
            continue;
        }
        loop {
            if cursor.goto_next_sibling() {
                break;
            }
            if !cursor.goto_parent() {
                return Array1::from_vec(counts.into_iter().map(|c| c as f64).collect());
            }
        }
    }
}

/// Get the vocabulary mapping: histogram index -> node kind name.
///
/// The returned Vec has the same length as the histogram. Index `i` corresponds
/// to the node kind with `kind_id == i`.
pub fn histogram_vocabulary(tree: &CodeTree) -> Vec<String> {
    let vocab = tree.language().node_kind_count();
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
    let mut cursor = tree.tree().walk();

    loop {
        if cursor.node().is_named() {
            count += 1;
        }
        if cursor.goto_first_child() {
            continue;
        }
        loop {
            if cursor.goto_next_sibling() {
                break;
            }
            if !cursor.goto_parent() {
                return count;
            }
        }
    }
}

/// Extract a dense adjacency matrix for named nodes only.
///
/// Rows and columns correspond to named nodes in pre-order traversal.
/// A non-zero entry `matrix[[i, j]] = 1.0` means node `i` is the parent of node `j`.
///
/// # Errors
/// Returns `CodeError::TreeTooLarge` if the tree has more than 1,000 named nodes.
/// Use `adjacency_graph()` (with the `graph` feature) for larger trees.
pub fn adjacency(tree: &CodeTree) -> Result<Array2<f64>, CodeError> {
    // First pass: count named nodes and assign indices
    let n = named_node_count(tree);
    if n > MAX_DENSE_NAMED_NODES {
        return Err(CodeError::TreeTooLarge(n, MAX_DENSE_NAMED_NODES));
    }
    if n == 0 {
        return Ok(Array2::zeros((0, 0)));
    }

    // Second pass: build index map (node_id -> index)
    let mut node_ids: Vec<usize> = Vec::with_capacity(n);
    let mut cursor = tree.tree().walk();

    loop {
        let node = cursor.node();
        if node.is_named() {
            node_ids.push(node.id());
        }
        if cursor.goto_first_child() {
            continue;
        }
        loop {
            if cursor.goto_next_sibling() {
                break;
            }
            if !cursor.goto_parent() {
                break;
            }
        }
        if node_ids.len() >= n {
            break;
        }
    }

    // Build id-to-index lookup
    let id_to_index: std::collections::HashMap<usize, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(idx, &id)| (id, idx))
        .collect();

    // Third pass: fill adjacency matrix
    let mut matrix = Array2::<f64>::zeros((n, n));
    let mut cursor = tree.tree().walk();
    fill_adjacency(&mut cursor, &id_to_index, &mut matrix);

    Ok(matrix)
}

fn fill_adjacency(
    cursor: &mut tree_sitter::TreeCursor,
    id_to_index: &std::collections::HashMap<usize, usize>,
    matrix: &mut Array2<f64>,
) {
    let parent = cursor.node();
    let parent_named = parent.is_named();
    let parent_idx = if parent_named {
        id_to_index.get(&parent.id()).copied()
    } else {
        None
    };

    if cursor.goto_first_child() {
        loop {
            let child = cursor.node();
            // If both parent and child are named, add an edge
            if let Some(pi) = parent_idx {
                if child.is_named() {
                    if let Some(&ci) = id_to_index.get(&child.id()) {
                        matrix[[pi, ci]] = 1.0;
                    }
                }
            }
            // Recurse — but we need to find the nearest named ancestor for unnamed parents
            fill_adjacency(cursor, id_to_index, matrix);

            if !cursor.goto_next_sibling() {
                break;
            }
        }
        cursor.goto_parent();
    }
}

/// Get node kind labels for the adjacency matrix rows/columns.
///
/// Returns the kind name for each named node in pre-order traversal order,
/// matching the row/column ordering of `adjacency()`.
pub fn adjacency_labels(tree: &CodeTree) -> Vec<String> {
    let mut labels = Vec::new();
    let mut cursor = tree.tree().walk();

    loop {
        let node = cursor.node();
        if node.is_named() {
            labels.push(node.kind().to_string());
        }
        if cursor.goto_first_child() {
            continue;
        }
        loop {
            if cursor.goto_next_sibling() {
                break;
            }
            if !cursor.goto_parent() {
                return labels;
            }
        }
    }
}

/// Extract a sparse adjacency graph for named nodes (any size tree).
///
/// Requires the `graph` feature flag. Returns a `machin_graph::Graph`
/// with directed edges from parent to child, weighted 1.0.
#[cfg(feature = "graph")]
pub fn adjacency_graph(tree: &CodeTree) -> machin_graph::Graph {
    use std::collections::HashMap;

    let mut node_ids: Vec<usize> = Vec::new();
    let mut cursor = tree.tree().walk();

    // Collect named node IDs in pre-order
    loop {
        if cursor.node().is_named() {
            node_ids.push(cursor.node().id());
        }
        if cursor.goto_first_child() {
            continue;
        }
        loop {
            if cursor.goto_next_sibling() {
                break;
            }
            if !cursor.goto_parent() {
                break;
            }
        }
        // Check if we've traversed the whole tree
        if cursor.node().id() == tree.tree().root_node().id()
            && !cursor.goto_first_child()
        {
            break;
        }
    }

    let id_to_index: HashMap<usize, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(idx, &id)| (id, idx))
        .collect();

    let n = node_ids.len();
    let mut adjacency: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();
    for i in 0..n {
        adjacency.insert(i, Vec::new());
    }

    // Walk again to build edges
    let mut cursor = tree.tree().walk();
    build_graph_edges(&mut cursor, &id_to_index, &mut adjacency);

    machin_graph::Graph {
        adjacency,
        node_count: n,
    }
}

#[cfg(feature = "graph")]
fn build_graph_edges(
    cursor: &mut tree_sitter::TreeCursor,
    id_to_index: &std::collections::HashMap<usize, usize>,
    adjacency: &mut std::collections::HashMap<usize, Vec<(usize, f64)>>,
) {
    let parent = cursor.node();
    let parent_idx = if parent.is_named() {
        id_to_index.get(&parent.id()).copied()
    } else {
        None
    };

    if cursor.goto_first_child() {
        loop {
            let child = cursor.node();
            if let Some(pi) = parent_idx {
                if child.is_named() {
                    if let Some(&ci) = id_to_index.get(&child.id()) {
                        adjacency.entry(pi).or_default().push((ci, 1.0));
                    }
                }
            }
            build_graph_edges(cursor, id_to_index, adjacency);
            if !cursor.goto_next_sibling() {
                break;
            }
        }
        cursor.goto_parent();
    }
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
}
