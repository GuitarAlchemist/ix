// Sample fixture file used by the ix-ai-annotations integration test.
// Each annotation below should round-trip through extract_workspace.

pub fn binary_search(arr: &[i32], target: i32) -> Option<usize> {
    // @ai:invariant arr is sorted ascending [T:test conf:0.95 src:test_search.rs:42]
    let mut lo = 0;
    let mut hi = arr.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        // @ai:assumption (hi - lo) does not overflow [P:assumed conf:0.7]
        match arr[mid].cmp(&target) {
            std::cmp::Ordering::Less => lo = mid + 1,
            std::cmp::Ordering::Greater => hi = mid,
            std::cmp::Ordering::Equal => return Some(mid),
        }
    }
    None
}

/// @ai:contract returns None iff target not present [T:manually-reviewed conf:0.85 src:PR#313]
pub fn search_api() {}

// @ai:hypothesis race-free under MIRI [U:uncertain conf:0.4]
// @ai:smell duplicate constants below [D:inferred conf:0.6]
fn _placeholder() {}
