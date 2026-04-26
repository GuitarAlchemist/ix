//! Agent routing: token-efficient skill dispatch using graph algorithms.
//!
//! Models Claude Code skills as a graph. Each skill has:
//! - An estimated token cost
//! - A capability vector (what it can do)
//! - Dependencies on other skills
//!
//! The router finds the optimal sequence of skills to fulfill a request
//! while minimizing total token consumption.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::graph::Graph;

/// A skill in the routing graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillNode {
    pub id: String,
    pub name: String,
    /// Estimated token cost for one invocation.
    pub token_cost: f64,
    /// Capability tags this skill provides.
    pub capabilities: Vec<String>,
    /// Skills that must run before this one.
    pub dependencies: Vec<String>,
    /// How relevant this skill is (0-1) for different query types.
    pub relevance_scores: HashMap<String, f64>,
}

/// The skill routing graph.
pub struct SkillRouter {
    pub skills: HashMap<String, SkillNode>,
    graph: Graph,
    id_to_index: HashMap<String, usize>,
    index_to_id: HashMap<usize, String>,
}

impl SkillRouter {
    pub fn new() -> Self {
        Self {
            skills: HashMap::new(),
            graph: Graph::new(),
            id_to_index: HashMap::new(),
            index_to_id: HashMap::new(),
        }
    }

    /// Register a skill.
    pub fn add_skill(&mut self, skill: SkillNode) {
        let idx = self.id_to_index.len();
        self.id_to_index.insert(skill.id.clone(), idx);
        self.index_to_id.insert(idx, skill.id.clone());
        self.skills.insert(skill.id.clone(), skill);
    }

    /// Build the dependency graph after all skills are added.
    pub fn build_graph(&mut self) {
        self.graph = Graph::with_nodes(self.skills.len());

        for skill in self.skills.values() {
            if let Some(&to_idx) = self.id_to_index.get(&skill.id) {
                for dep in &skill.dependencies {
                    if let Some(&from_idx) = self.id_to_index.get(dep) {
                        // Edge from dependency to dependent, weighted by token cost
                        self.graph.add_edge(from_idx, to_idx, skill.token_cost);
                    }
                }
            }
        }
    }

    /// Find the execution order for skills that provide the requested capabilities.
    /// Returns skills in dependency-respecting order, minimizing total token cost.
    pub fn route(&self, required_capabilities: &[String], token_budget: f64) -> RoutePlan {
        // 1. Find skills that match required capabilities
        let mut needed_skills: Vec<&SkillNode> = Vec::new();
        let mut covered_caps: Vec<String> = Vec::new();

        // Greedy: for each capability, pick the cheapest skill that provides it
        for cap in required_capabilities {
            if covered_caps.contains(cap) {
                continue;
            }

            let best = self
                .skills
                .values()
                .filter(|s| s.capabilities.contains(cap))
                .min_by(|a, b| a.token_cost.partial_cmp(&b.token_cost).unwrap());

            if let Some(skill) = best {
                needed_skills.push(skill);
                covered_caps.extend(skill.capabilities.iter().cloned());
            }
        }

        // 2. Add dependencies
        let mut all_skills: Vec<String> = needed_skills.iter().map(|s| s.id.clone()).collect();
        let mut i = 0;
        while i < all_skills.len() {
            if let Some(skill) = self.skills.get(&all_skills[i]) {
                for dep in &skill.dependencies {
                    if !all_skills.contains(dep) {
                        all_skills.push(dep.clone());
                    }
                }
            }
            i += 1;
        }

        // 3. Topological sort for execution order
        let execution_order = self.topo_sort_subset(&all_skills);

        // 4. Compute total cost and check budget
        let total_cost: f64 = execution_order
            .iter()
            .filter_map(|id| self.skills.get(id))
            .map(|s| s.token_cost)
            .sum();

        let uncovered: Vec<String> = required_capabilities
            .iter()
            .filter(|cap| !covered_caps.contains(cap))
            .cloned()
            .collect();

        RoutePlan {
            execution_order,
            total_token_cost: total_cost,
            within_budget: total_cost <= token_budget,
            uncovered_capabilities: uncovered,
        }
    }

    /// Rank skills by relevance to a query type.
    pub fn rank_skills(&self, query_type: &str, top_k: usize) -> Vec<(String, f64)> {
        let mut scored: Vec<(String, f64)> = self
            .skills
            .values()
            .map(|s| {
                let score = s.relevance_scores.get(query_type).copied().unwrap_or(0.0);
                (s.id.clone(), score)
            })
            .collect();
        scored.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
        scored.truncate(top_k);
        scored
    }

    fn topo_sort_subset(&self, subset: &[String]) -> Vec<String> {
        let subset_set: std::collections::HashSet<&String> = subset.iter().collect();
        let mut in_degree: HashMap<&String, usize> = HashMap::new();

        for id in subset {
            in_degree.insert(id, 0);
        }

        for id in subset {
            if let Some(skill) = self.skills.get(id) {
                for dep in &skill.dependencies {
                    if subset_set.contains(dep) {
                        *in_degree.entry(id).or_insert(0) += 1;
                    }
                }
            }
        }

        let mut queue: std::collections::VecDeque<String> = in_degree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&id, _)| id.clone())
            .collect();
        let mut order = Vec::new();

        while let Some(id) = queue.pop_front() {
            order.push(id.clone());
            // Find skills that depend on this one
            for other_id in subset {
                if let Some(skill) = self.skills.get(other_id) {
                    if skill.dependencies.contains(&id) {
                        if let Some(d) = in_degree.get_mut(other_id) {
                            *d -= 1;
                            if *d == 0 {
                                queue.push_back(other_id.clone());
                            }
                        }
                    }
                }
            }
        }

        order
    }
}

impl Default for SkillRouter {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a routing query.
#[derive(Debug, Clone)]
pub struct RoutePlan {
    /// Skills to execute, in order.
    pub execution_order: Vec<String>,
    /// Estimated total token cost.
    pub total_token_cost: f64,
    /// Whether the plan fits within the token budget.
    pub within_budget: bool,
    /// Capabilities that no skill could provide.
    pub uncovered_capabilities: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_router() -> SkillRouter {
        let mut router = SkillRouter::new();

        router.add_skill(SkillNode {
            id: "data-load".into(),
            name: "Data Loader".into(),
            token_cost: 100.0,
            capabilities: vec!["load-csv".into(), "load-json".into()],
            dependencies: vec![],
            relevance_scores: [("data".into(), 0.9), ("ml".into(), 0.3)].into(),
        });

        router.add_skill(SkillNode {
            id: "preprocess".into(),
            name: "Preprocessor".into(),
            token_cost: 200.0,
            capabilities: vec!["normalize".into(), "feature-select".into()],
            dependencies: vec!["data-load".into()],
            relevance_scores: [("data".into(), 0.7), ("ml".into(), 0.5)].into(),
        });

        router.add_skill(SkillNode {
            id: "train".into(),
            name: "Model Trainer".into(),
            token_cost: 500.0,
            capabilities: vec!["train-model".into(), "evaluate".into()],
            dependencies: vec!["preprocess".into()],
            relevance_scores: [("ml".into(), 0.95)].into(),
        });

        router.add_skill(SkillNode {
            id: "visualize".into(),
            name: "Visualizer".into(),
            token_cost: 150.0,
            capabilities: vec!["plot".into(), "chart".into()],
            dependencies: vec![],
            relevance_scores: [("data".into(), 0.8), ("ml".into(), 0.4)].into(),
        });

        router.build_graph();
        router
    }

    #[test]
    fn test_route_simple() {
        let router = build_test_router();
        let plan = router.route(&["train-model".into()], 1000.0);

        // Should include data-load -> preprocess -> train
        assert_eq!(plan.execution_order.len(), 3);
        assert!(plan.within_budget);
        assert!(plan.uncovered_capabilities.is_empty());

        // Check order: data-load before preprocess, preprocess before train
        let load_pos = plan
            .execution_order
            .iter()
            .position(|s| s == "data-load")
            .unwrap();
        let prep_pos = plan
            .execution_order
            .iter()
            .position(|s| s == "preprocess")
            .unwrap();
        let train_pos = plan
            .execution_order
            .iter()
            .position(|s| s == "train")
            .unwrap();
        assert!(load_pos < prep_pos);
        assert!(prep_pos < train_pos);
    }

    #[test]
    fn test_route_budget_exceeded() {
        let router = build_test_router();
        let plan = router.route(&["train-model".into()], 500.0);
        // data-load(100) + preprocess(200) + train(500) = 800 > 500
        assert!(!plan.within_budget);
    }

    #[test]
    fn test_rank_skills() {
        let router = build_test_router();
        let ranked = router.rank_skills("ml", 2);
        assert_eq!(ranked[0].0, "train"); // Most relevant for ML
    }

    #[test]
    fn test_uncovered_capability() {
        let router = build_test_router();
        let plan = router.route(&["deploy".into()], 10000.0);
        assert!(plan.uncovered_capabilities.contains(&"deploy".to_string()));
    }
}
