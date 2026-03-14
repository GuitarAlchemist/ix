//! LRU (Least Recently Used) eviction policy.
//!
//! Doubly-linked list + hash map for O(1) access, insert, and eviction.

use std::collections::HashMap;

/// Node in the LRU doubly-linked list.
struct Node {
    key: String,
    prev: Option<usize>,
    next: Option<usize>,
}

/// LRU eviction tracker.
///
/// Maintains access order: most recent at the tail, least recent at the head.
pub struct LruPolicy {
    nodes: Vec<Node>,
    index: HashMap<String, usize>,
    head: Option<usize>, // Least recently used
    tail: Option<usize>, // Most recently used
    free_slots: Vec<usize>,
}

impl Default for LruPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl LruPolicy {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            index: HashMap::new(),
            head: None,
            tail: None,
            free_slots: Vec::new(),
        }
    }

    /// Mark a key as recently used. Adds it if not present.
    pub fn touch(&mut self, key: &str) {
        if let Some(&idx) = self.index.get(key) {
            // Move to tail (most recently used)
            self.detach(idx);
            self.attach_tail(idx);
        } else {
            // Insert new node
            let node = Node {
                key: key.to_string(),
                prev: None,
                next: None,
            };

            let idx = if let Some(free) = self.free_slots.pop() {
                self.nodes[free] = node;
                free
            } else {
                self.nodes.push(node);
                self.nodes.len() - 1
            };

            self.index.insert(key.to_string(), idx);
            self.attach_tail(idx);
        }
    }

    /// Remove a key from the LRU tracker.
    pub fn remove(&mut self, key: &str) {
        if let Some(idx) = self.index.remove(key) {
            self.detach(idx);
            self.free_slots.push(idx);
        }
    }

    /// Evict the least recently used key. Returns the evicted key.
    pub fn evict(&mut self) -> Option<String> {
        let head_idx = self.head?;
        let key = self.nodes[head_idx].key.clone();
        self.remove(&key);
        Some(key)
    }

    /// Peek at the least recently used key without removing it.
    pub fn peek_lru(&self) -> Option<&str> {
        self.head.map(|idx| self.nodes[idx].key.as_str())
    }

    /// Number of tracked keys.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.index.clear();
        self.head = None;
        self.tail = None;
        self.free_slots.clear();
    }

    /// Detach a node from the linked list.
    fn detach(&mut self, idx: usize) {
        let prev = self.nodes[idx].prev;
        let next = self.nodes[idx].next;

        if let Some(p) = prev {
            self.nodes[p].next = next;
        } else {
            self.head = next;
        }

        if let Some(n) = next {
            self.nodes[n].prev = prev;
        } else {
            self.tail = prev;
        }

        self.nodes[idx].prev = None;
        self.nodes[idx].next = None;
    }

    /// Attach a node at the tail (most recently used).
    fn attach_tail(&mut self, idx: usize) {
        self.nodes[idx].prev = self.tail;
        self.nodes[idx].next = None;

        if let Some(t) = self.tail {
            self.nodes[t].next = Some(idx);
        }

        self.tail = Some(idx);

        if self.head.is_none() {
            self.head = Some(idx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_basic() {
        let mut lru = LruPolicy::new();
        lru.touch("a");
        lru.touch("b");
        lru.touch("c");

        assert_eq!(lru.len(), 3);
        assert_eq!(lru.peek_lru(), Some("a"));
    }

    #[test]
    fn test_lru_eviction_order() {
        let mut lru = LruPolicy::new();
        lru.touch("a");
        lru.touch("b");
        lru.touch("c");

        assert_eq!(lru.evict(), Some("a".to_string()));
        assert_eq!(lru.evict(), Some("b".to_string()));
        assert_eq!(lru.evict(), Some("c".to_string()));
        assert_eq!(lru.evict(), None);
    }

    #[test]
    fn test_lru_touch_moves_to_back() {
        let mut lru = LruPolicy::new();
        lru.touch("a");
        lru.touch("b");
        lru.touch("c");

        // Touch "a" again — should move it to back
        lru.touch("a");

        assert_eq!(lru.evict(), Some("b".to_string()));
        assert_eq!(lru.evict(), Some("c".to_string()));
        assert_eq!(lru.evict(), Some("a".to_string()));
    }

    #[test]
    fn test_lru_remove() {
        let mut lru = LruPolicy::new();
        lru.touch("a");
        lru.touch("b");
        lru.touch("c");

        lru.remove("b");
        assert_eq!(lru.len(), 2);

        assert_eq!(lru.evict(), Some("a".to_string()));
        assert_eq!(lru.evict(), Some("c".to_string()));
    }
}
