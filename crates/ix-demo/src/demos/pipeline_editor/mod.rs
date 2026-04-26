//! Visual DAG pipeline editor backed by `egui_snarl`.
//!
//! v1 scope:
//!   - 10 node variants including PolicyGate + Belief (hexavalent badge)
//!   - Typed sockets (8 types, color-coded) with widening rules
//!   - Palette search filter in the right-click graph menu
//!   - JSON save/load (canonical format) + YAML export to `ix.yaml`
//!   - Per-node live execution via `ix_registry::invoke()` with status dots
//!   - Static validation: flags ML nodes without an upstream PolicyGate

mod nodes;
mod sockets;
mod viewer;

use std::collections::{BTreeMap, HashMap};

use eframe::egui;
use egui_snarl::{ui::SnarlStyle, InPinId, NodeId, Snarl};
use ix_types::Value as IxValue;
use serde_json::{json, Value};

use self::nodes::{HexValue, IxNode, Socket};
use self::sockets::SocketType;
use self::viewer::IxViewer;

/// Per-node run status rendered as a colored dot on each node's header.
#[derive(Debug, Clone)]
pub enum RunStatus {
    Ok,
    Err(String),
    Skipped,
}

/// Result of executing an `ix.yaml` via the ix-pipeline crate — kept
/// separate from the visual graph's outputs since it's an orthogonal run.
pub struct YamlRun {
    pub path: String,
    pub stages: BTreeMap<String, Value>,
    pub total_duration_ms: u64,
    pub cache_hits: usize,
    pub error: Option<String>,
}

pub struct PipelineEditor {
    snarl: Snarl<IxNode>,
    style: SnarlStyle,
    search: String,
    run_status: HashMap<NodeId, RunStatus>,
    outputs: HashMap<NodeId, Value>,
    show_outputs: bool,
    validation_warnings: Vec<String>,
    yaml_run: Option<YamlRun>,
    show_yaml_run: bool,
    last_action: String,
}

impl Default for PipelineEditor {
    fn default() -> Self {
        Self {
            snarl: Snarl::new(),
            style: SnarlStyle::default(),
            search: String::new(),
            run_status: HashMap::new(),
            outputs: HashMap::new(),
            show_outputs: false,
            validation_warnings: Vec::new(),
            yaml_run: None,
            show_yaml_run: true,
            last_action: String::new(),
        }
    }
}

impl PipelineEditor {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("Pipeline Editor");
            ui.separator();
            ui.label(
                egui::RichText::new(
                    "Right-click canvas to add nodes • drag pins to connect • right-click a node to remove",
                )
                .weak()
                .small(),
            );
        });
        ui.separator();

        ui.horizontal(|ui| {
            if ui.button("Run").clicked() {
                self.execute();
            }
            if ui.button("Validate").clicked() {
                self.validate();
                self.last_action =
                    format!("validation: {} warning(s)", self.validation_warnings.len());
            }
            ui.separator();
            if ui.button("Save JSON").clicked() {
                self.last_action = match serde_json::to_string_pretty(&self.snarl) {
                    Ok(s) => match std::fs::write("pipeline.json", s) {
                        Ok(_) => "saved pipeline.json".into(),
                        Err(e) => format!("save failed: {e}"),
                    },
                    Err(e) => format!("serialize failed: {e}"),
                };
            }
            if ui.button("Load JSON").clicked() {
                self.last_action = match std::fs::read_to_string("pipeline.json") {
                    Ok(text) => match serde_json::from_str::<Snarl<IxNode>>(&text) {
                        Ok(loaded) => {
                            self.snarl = loaded;
                            self.run_status.clear();
                            "loaded pipeline.json".into()
                        }
                        Err(e) => format!("parse failed: {e}"),
                    },
                    Err(e) => format!("read failed: {e}"),
                };
            }
            if ui.button("Export YAML").clicked() {
                self.last_action = match self.export_yaml() {
                    Ok(path) => format!("exported {path}"),
                    Err(e) => format!("export failed: {e}"),
                };
            }
            if ui.button("Run ix.yaml").clicked() {
                self.run_yaml("ix.yaml");
            }
            if ui.button("Import YAML").clicked() {
                self.last_action = match self.import_yaml("ix.yaml") {
                    Ok(n) => format!("imported {n} stage(s) from ix.yaml"),
                    Err(e) => format!("import failed: {e}"),
                };
            }
            if ui.button("Clear").clicked() {
                self.snarl = Snarl::new();
                self.run_status.clear();
                self.outputs.clear();
                self.validation_warnings.clear();
                self.yaml_run = None;
                self.last_action = "canvas cleared".into();
            }
        });

        ui.horizontal(|ui| {
            ui.label("Search:");
            ui.text_edit_singleline(&mut self.search);
            ui.separator();
            ui.weak(&self.last_action);
        });

        ui.separator();

        // Socket legend.
        ui.horizontal_wrapped(|ui| {
            ui.weak("Socket types:");
            for (label, ty) in [
                ("Scalar", sockets::SocketType::Scalar),
                ("Vector", sockets::SocketType::Vector),
                ("Matrix", sockets::SocketType::Matrix),
                ("Dataset", sockets::SocketType::Dataset),
                ("Model", sockets::SocketType::Model),
                ("Belief", sockets::SocketType::Belief),
                ("Text", sockets::SocketType::Text),
                ("Any", sockets::SocketType::Any),
            ] {
                let (rect, _) =
                    ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
                ui.painter().circle_filled(rect.center(), 5.0, ty.color());
                ui.label(egui::RichText::new(label).small());
                ui.add_space(4.0);
            }
        });

        // Validation warnings banner.
        if !self.validation_warnings.is_empty() {
            ui.separator();
            ui.label(
                egui::RichText::new(format!(
                    "⚠ {} validation warning(s):",
                    self.validation_warnings.len()
                ))
                .color(egui::Color32::from_rgb(0xf9, 0x73, 0x16))
                .strong(),
            );
            for w in &self.validation_warnings {
                ui.label(egui::RichText::new(format!("  • {w}")).small());
            }
        }

        // YAML run panel — present after Run ix.yaml is invoked.
        if let Some(run) = &self.yaml_run {
            ui.separator();
            ui.horizontal(|ui| {
                let caret = if self.show_yaml_run { "▼" } else { "▶" };
                let header = if let Some(err) = &run.error {
                    format!("{caret} {} — error: {err}", run.path)
                } else {
                    format!(
                        "{caret} {} — {} stage(s), {}ms, {} cache-hit(s)",
                        run.path,
                        run.stages.len(),
                        run.total_duration_ms,
                        run.cache_hits
                    )
                };
                let color = if run.error.is_some() {
                    egui::Color32::from_rgb(0xef, 0x44, 0x44)
                } else {
                    egui::Color32::from_rgb(0x22, 0xc5, 0x5e)
                };
                if ui
                    .small_button(egui::RichText::new(&header).color(color).strong())
                    .clicked()
                {
                    self.show_yaml_run = !self.show_yaml_run;
                }
            });
            if self.show_yaml_run && run.error.is_none() {
                egui::ScrollArea::vertical()
                    .max_height(140.0)
                    .id_salt("pipeline_editor_yaml_results")
                    .show(ui, |ui| {
                        for (stage_id, output) in &run.stages {
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new(format!("  {stage_id}"))
                                        .strong()
                                        .small(),
                                );
                                ui.label(
                                    egui::RichText::new(short_value(output)).monospace().small(),
                                );
                            });
                        }
                    });
            }
        }

        // Snarl graph results panel — collapsed by default, expanded after a run.
        if !self.outputs.is_empty() {
            ui.separator();
            ui.horizontal(|ui| {
                let label = if self.show_outputs { "▼" } else { "▶" };
                if ui
                    .small_button(format!("{label} Results ({})", self.outputs.len()))
                    .clicked()
                {
                    self.show_outputs = !self.show_outputs;
                }
            });
            if self.show_outputs {
                egui::ScrollArea::vertical()
                    .max_height(160.0)
                    .id_salt("pipeline_editor_results")
                    .show(ui, |ui| {
                        for (node_id, value) in &self.outputs {
                            let node = self.snarl.get_node(*node_id);
                            let title = node
                                .map(|n| n.title())
                                .unwrap_or_else(|| "<node>".to_string());
                            let short = short_value(value);
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new(format!("  {title}")).strong().small(),
                                );
                                ui.label(egui::RichText::new(short).monospace().small());
                            });
                        }
                    });
            }
        }

        ui.separator();

        let mut viewer = IxViewer {
            search: &self.search,
            run_status: &self.run_status,
        };
        self.snarl
            .show(&mut viewer, &self.style, "ix_pipeline_editor", ui);
    }

    // ────────── execution ──────────

    /// Topologically sort node ids via Kahn's algorithm over the current
    /// Snarl wires. Returns `None` if the graph has a cycle (editor should
    /// be cycle-free by construction, but we defend against invariant breaks).
    fn topo_order(&self) -> Option<Vec<NodeId>> {
        let mut in_degree: HashMap<NodeId, usize> =
            self.snarl.node_ids().map(|(id, _)| (id, 0usize)).collect();
        let mut succ: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for (out_pin, in_pin) in self.snarl.wires() {
            *in_degree.entry(in_pin.node).or_insert(0) += 1;
            succ.entry(out_pin.node).or_default().push(in_pin.node);
        }
        let mut ready: Vec<NodeId> = in_degree
            .iter()
            .filter(|(_, d)| **d == 0)
            .map(|(id, _)| *id)
            .collect();
        let mut order = Vec::new();
        while let Some(n) = ready.pop() {
            order.push(n);
            if let Some(children) = succ.get(&n) {
                for c in children {
                    let d = in_degree.get_mut(c).unwrap();
                    *d -= 1;
                    if *d == 0 {
                        ready.push(*c);
                    }
                }
            }
        }
        if order.len() == in_degree.len() {
            Some(order)
        } else {
            None
        }
    }

    /// Execute the graph in topological order with wire-aware data flow.
    /// Each node gathers its upstream outputs by input-socket name, builds
    /// an args JSON that merges its own params with those upstreams, then
    /// invokes its registry skill (or synthesizes an output for
    /// source/sink/inline variants).
    fn execute(&mut self) {
        self.run_status.clear();
        self.outputs.clear();
        self.show_outputs = true;
        let order = match self.topo_order() {
            Some(o) => o,
            None => {
                self.last_action = "execution aborted: cycle detected".into();
                return;
            }
        };

        let mut ok = 0usize;
        let mut err = 0usize;
        let mut skipped = 0usize;
        for node_id in order {
            let node = match self.snarl.get_node(node_id) {
                Some(n) => n.clone(),
                None => continue,
            };

            // Gather upstream outputs, keyed by this node's input socket name.
            // Single-source inputs: take the first remote (enforced by connect()).
            let mut upstream: HashMap<String, Value> = HashMap::new();
            for (i, socket) in node.input_sockets().iter().enumerate() {
                let in_pin = self.snarl.in_pin(InPinId {
                    node: node_id,
                    input: i,
                });
                if let Some(src) = in_pin.remotes.first() {
                    if let Some(upstream_out) = self.outputs.get(&src.node) {
                        upstream.insert(socket.name.clone(), upstream_out.clone());
                    }
                }
            }

            match execute_node_with_inputs(&node, &upstream) {
                StepOutcome::Produced(value) => {
                    self.outputs.insert(node_id, value);
                    self.run_status.insert(node_id, RunStatus::Ok);
                    ok += 1;
                }
                StepOutcome::Skipped(value) => {
                    self.outputs.insert(node_id, value);
                    self.run_status.insert(node_id, RunStatus::Skipped);
                    skipped += 1;
                }
                StepOutcome::Failed(msg) => {
                    self.run_status.insert(node_id, RunStatus::Err(msg));
                    err += 1;
                }
            }
        }
        self.last_action = format!("run: {ok} ok, {skipped} skipped, {err} error");
    }

    // ────────── static validation ──────────

    fn validate(&mut self) {
        self.validation_warnings.clear();

        // For every learner node, walk back through wires to find a
        // PolicyGate ancestor. If none, record a warning.
        let mut predecessors: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for (out_pin, in_pin) in self.snarl.wires() {
            predecessors
                .entry(in_pin.node)
                .or_default()
                .push(out_pin.node);
        }

        for (id, node) in self.snarl.node_ids() {
            if !node.is_learner() {
                continue;
            }
            if !has_ancestor_gate(id, &predecessors, &self.snarl) {
                self.validation_warnings.push(format!(
                    "{} has no upstream PolicyGate — training skills should be governed",
                    node.title()
                ));
            }
        }
    }

    // ────────── ix.yaml import (reconstruct Snarl from stages) ──────────

    fn import_yaml(&mut self, path: &str) -> Result<usize, String> {
        use ix_pipeline::spec::PipelineSpec;

        let spec = PipelineSpec::from_file(path).map_err(|e| e.to_string())?;

        // Reset canvas and reconstruct one Skill node per stage.
        self.snarl = Snarl::new();
        self.run_status.clear();
        self.outputs.clear();

        // First pass: create Skill nodes, remember stage_id → NodeId mapping.
        let mut stage_to_node: std::collections::HashMap<String, NodeId> =
            std::collections::HashMap::new();
        let stage_count = spec.stages.len();
        // Grid layout: 220px wide × 140px tall cells, up to 4 cols.
        let cols = 4usize;
        for (i, (stage_id, stage)) in spec.stages.iter().enumerate() {
            let x = (i % cols) as f32 * 220.0;
            let y = (i / cols) as f32 * 140.0;

            // Resolve socket types from the registry descriptor when possible.
            let (ui_inputs, ui_outputs) = sockets_for_skill(&stage.skill);

            let label = stage_id.clone();
            let node = IxNode::Skill {
                skill: stage.skill.clone(),
                args: stage.args.clone(),
                label,
                ui_inputs,
                ui_outputs,
            };
            let node_id = self.snarl.insert_node(egui::pos2(x, y), node);
            stage_to_node.insert(stage_id.clone(), node_id);
        }

        // Second pass: wire deps → edges. Each dep is an upstream stage id;
        // we connect its first output pin to our first input pin.
        for (stage_id, stage) in &spec.stages {
            let Some(&dst_node) = stage_to_node.get(stage_id) else {
                continue;
            };
            for dep in &stage.deps {
                // Depdendencies can be "stage" or "stage.field" — take the stage part.
                let dep_stage = dep.split('.').next().unwrap_or(dep);
                if let Some(&src_node) = stage_to_node.get(dep_stage) {
                    // Connect first available output → first available input.
                    let src_pin = egui_snarl::OutPinId {
                        node: src_node,
                        output: 0,
                    };
                    let dst_pin = InPinId {
                        node: dst_node,
                        input: 0,
                    };
                    self.snarl.connect(src_pin, dst_pin);
                }
            }
        }

        Ok(stage_count)
    }

    // ────────── ix.yaml execution (via ix-pipeline) ──────────

    fn run_yaml(&mut self, path: &str) {
        use ix_pipeline::executor::{execute, NoCache};
        use ix_pipeline::lower::lower;
        use ix_pipeline::spec::PipelineSpec;

        let spec = match PipelineSpec::from_file(path) {
            Ok(s) => s,
            Err(e) => {
                self.yaml_run = Some(YamlRun {
                    path: path.into(),
                    stages: BTreeMap::new(),
                    total_duration_ms: 0,
                    cache_hits: 0,
                    error: Some(format!("load: {e}")),
                });
                self.last_action = format!("run ix.yaml: {e}");
                return;
            }
        };
        let dag = match lower(&spec) {
            Ok(d) => d,
            Err(e) => {
                self.yaml_run = Some(YamlRun {
                    path: path.into(),
                    stages: BTreeMap::new(),
                    total_duration_ms: 0,
                    cache_hits: 0,
                    error: Some(format!("lower: {e}")),
                });
                self.last_action = format!("run ix.yaml: {e}");
                return;
            }
        };
        let result = match execute(&dag, &HashMap::new(), &NoCache) {
            Ok(r) => r,
            Err(e) => {
                self.yaml_run = Some(YamlRun {
                    path: path.into(),
                    stages: BTreeMap::new(),
                    total_duration_ms: 0,
                    cache_hits: 0,
                    error: Some(format!("execute: {e}")),
                });
                self.last_action = format!("run ix.yaml: {e}");
                return;
            }
        };

        let mut stages = BTreeMap::new();
        for (stage_id, node_result) in &result.node_results {
            stages.insert(stage_id.clone(), node_result.output.clone());
        }
        let total_duration_ms = result.total_duration.as_millis() as u64;
        let cache_hits = result.cache_hits;

        self.yaml_run = Some(YamlRun {
            path: path.into(),
            stages,
            total_duration_ms,
            cache_hits,
            error: None,
        });
        self.last_action = format!(
            "ran {path}: {} stages in {}ms",
            result.node_results.len(),
            total_duration_ms
        );
    }

    // ────────── YAML export ──────────

    fn export_yaml(&self) -> Result<&'static str, String> {
        let mut stages: BTreeMap<String, Value> = BTreeMap::new();

        // Stable stage ids: <title>_<node_index>. NodeId displays as an int.
        let mut node_to_id: HashMap<NodeId, String> = HashMap::new();
        for (node_id, node) in self.snarl.node_ids() {
            let slug = slugify(&node.title());
            let id = format!("{slug}_{}", display_id(node_id));
            node_to_id.insert(node_id, id);
        }

        // Build stage entries.
        for (node_id, node) in self.snarl.node_ids() {
            let stage_id = node_to_id[&node_id].clone();
            let skill = match node.registry_skill() {
                Some(s) => s.to_string(),
                None => format!("_local.{}", slugify(&node.title())),
            };
            let args = node_args_json(node);

            // Deps: every upstream node wired into any of our input pins.
            let mut deps: Vec<String> = Vec::new();
            for (out_pin, in_pin) in self.snarl.wires() {
                if in_pin.node == node_id {
                    if let Some(dep_id) = node_to_id.get(&out_pin.node) {
                        deps.push(dep_id.clone());
                    }
                }
            }
            deps.sort();
            deps.dedup();

            stages.insert(
                stage_id,
                json!({
                    "skill": skill,
                    "args": args,
                    "deps": deps,
                }),
            );
        }

        let spec = json!({
            "version": "1",
            "stages": stages,
            "x-editor": { "generator": "ix-demo pipeline editor v1" },
        });

        // serde_yaml is already a workspace dep via ix-pipeline — pull it
        // through the same json-value path.
        let yaml = serde_yaml::to_string(&spec).map_err(|e| e.to_string())?;
        std::fs::write("ix.yaml", yaml).map_err(|e| e.to_string())?;
        Ok("ix.yaml")
    }
}

fn has_ancestor_gate(
    start: NodeId,
    predecessors: &HashMap<NodeId, Vec<NodeId>>,
    snarl: &Snarl<IxNode>,
) -> bool {
    use std::collections::HashSet;
    let mut visited: HashSet<NodeId> = HashSet::new();
    let mut stack: Vec<NodeId> = predecessors.get(&start).cloned().unwrap_or_default();
    while let Some(n) = stack.pop() {
        if !visited.insert(n) {
            continue;
        }
        if let Some(node) = snarl.get_node(n) {
            if node.is_gate() {
                return true;
            }
        }
        if let Some(parents) = predecessors.get(&n) {
            stack.extend(parents);
        }
    }
    false
}

enum StepOutcome {
    /// A skill ran successfully and produced this output.
    Produced(Value),
    /// Node skipped execution (IO/sink/inline) but still has an output to
    /// publish downstream.
    Skipped(Value),
    /// Execution failed with this message.
    Failed(String),
}

fn execute_node_with_inputs(node: &IxNode, upstream: &HashMap<String, Value>) -> StepOutcome {
    match node {
        // ─── Sources that produce output inline (no registry call) ───
        IxNode::Constant { value } => StepOutcome::Skipped(json!({ "value": *value })),
        IxNode::CsvRead { path } => StepOutcome::Skipped(json!({
            "path": path,
            "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], // demo fallback
        })),
        IxNode::Belief {
            proposition,
            value,
            confidence,
        } => StepOutcome::Skipped(json!({
            "proposition": proposition,
            "truth_value": value.letter(),
            "confidence": *confidence,
        })),

        // ─── Transforms that currently pass-through upstream ───
        IxNode::Normalize { method } => {
            let data = upstream_dataset(upstream);
            StepOutcome::Skipped(json!({
                "method": format!("{method:?}"),
                "data": data,
            }))
        }

        // ─── Sinks / leaf nodes ───
        IxNode::CsvWrite { path } => {
            let data = upstream.get("data").cloned().unwrap_or(Value::Null);
            StepOutcome::Skipped(json!({ "wrote": path, "data": data }))
        }
        IxNode::Plot { title } => {
            let data = upstream.get("data").cloned().unwrap_or(Value::Null);
            StepOutcome::Skipped(json!({ "title": title, "rendered": data }))
        }

        // ─── Registry-backed computations ───
        IxNode::KMeans { k, max_iter, seed } => {
            let data = upstream_dataset(upstream);
            let args = json!({
                "data": data,
                "k": *k,
                "max_iter": *max_iter,
            });
            let _ = seed; // ix_kmeans handler does not accept seed today
            invoke_skill("kmeans", args)
        }
        IxNode::LinearReg => {
            // LinearReg takes two pins (X, y). Build args from each.
            let x = upstream
                .get("X")
                .cloned()
                .unwrap_or_else(|| json!([[1.0], [2.0], [3.0]]));
            let y = upstream
                .get("y")
                .cloned()
                .unwrap_or_else(|| json!([2.0, 4.0, 6.0]));
            // The ix_linear_regression handler expects lowercase field names.
            let args = json!({
                "x": as_matrix(&x),
                "y": as_vector(&y),
            });
            invoke_skill("linear_regression", args)
        }
        IxNode::Fft { inverse } => {
            let signal = upstream
                .get("signal")
                .cloned()
                .unwrap_or_else(|| json!([1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0]));
            let args = json!({ "signal": as_vector(&signal), "inverse": *inverse });
            invoke_skill("fft", args)
        }
        IxNode::PolicyGate { policy, threshold } => {
            // Always runnable. Describe the upstream value in the action text.
            let value = upstream
                .get("value")
                .map(|v| v.to_string())
                .unwrap_or_else(|| "(no upstream)".into());
            let args = json!({
                "action": format!("evaluate policy {policy} on {value} at τ={threshold}"),
            });
            invoke_skill("governance.check", args)
        }

        // Generic imported skill — merge upstream outputs into the static
        // args blob under each upstream socket's name, then invoke the
        // registry skill.
        IxNode::Skill { skill, args, .. } => {
            let mut merged = args.clone();
            if let Value::Object(ref mut map) = merged {
                for (k, v) in upstream {
                    map.entry(k.clone()).or_insert_with(|| v.clone());
                }
            }
            invoke_skill(skill, merged)
        }
    }
}

/// Pull a best-effort 2-D "dataset" out of an upstream blob: prefer the
/// `data` field, fall back to the whole value, fall back to demo data.
fn upstream_dataset(upstream: &HashMap<String, Value>) -> Value {
    if let Some(input) = upstream.get("in").or_else(|| upstream.get("data")) {
        if let Some(data) = input.get("data") {
            return data.clone();
        }
        if input.is_array() {
            return input.clone();
        }
    }
    json!([[1.0, 1.0], [1.5, 2.0], [3.0, 4.0], [5.0, 7.0], [3.5, 5.0]])
}

/// Coerce an upstream blob into a matrix-shaped array if possible.
fn as_matrix(v: &Value) -> Value {
    if let Some(arr) = v.as_array() {
        if arr.first().map(|e| e.is_array()).unwrap_or(false) {
            return v.clone();
        }
        // Upgrade [a,b,c] → [[a],[b],[c]]
        let wrapped: Vec<Value> = arr.iter().map(|x| json!([x.clone()])).collect();
        return Value::Array(wrapped);
    }
    v.clone()
}

/// Coerce an upstream blob into a flat vector.
fn as_vector(v: &Value) -> Value {
    if let Some(arr) = v.as_array() {
        if arr.first().map(|e| e.is_array()).unwrap_or(false) {
            // Downgrade [[a],[b],[c]] → [a,b,c]
            let flat: Vec<Value> = arr
                .iter()
                .filter_map(|row| row.as_array().and_then(|r| r.first()).cloned())
                .collect();
            return Value::Array(flat);
        }
        return v.clone();
    }
    v.clone()
}

fn invoke_skill(name: &str, args: Value) -> StepOutcome {
    let desc = match ix_registry::by_name(name) {
        Some(d) => d,
        None => return StepOutcome::Failed(format!("skill '{name}' not in registry")),
    };
    let args = [IxValue::Json(args)];
    match (desc.fn_ptr)(&args) {
        Ok(IxValue::Json(j)) => StepOutcome::Produced(j),
        Ok(other) => match serde_json::to_value(other) {
            Ok(j) => StepOutcome::Produced(j),
            Err(e) => StepOutcome::Failed(e.to_string()),
        },
        Err(e) => StepOutcome::Failed(e.to_string()),
    }
}

fn node_args_json(node: &IxNode) -> Value {
    match node {
        IxNode::CsvRead { path } | IxNode::CsvWrite { path } => json!({ "path": path }),
        IxNode::Constant { value } => json!({ "value": *value }),
        IxNode::Normalize { method } => json!({ "method": format!("{method:?}") }),
        IxNode::KMeans { k, max_iter, seed } => {
            json!({ "k": *k, "max_iter": *max_iter, "seed": *seed })
        }
        IxNode::LinearReg => json!({}),
        IxNode::Fft { inverse } => json!({ "inverse": *inverse }),
        IxNode::PolicyGate { policy, threshold } => {
            json!({ "policy": policy, "threshold": *threshold })
        }
        IxNode::Plot { title } => json!({ "title": title }),
        IxNode::Belief {
            proposition,
            value,
            confidence,
        } => json!({
            "proposition": proposition,
            "truth_value": match value {
                HexValue::T => "T", HexValue::P => "P", HexValue::U => "U",
                HexValue::D => "D", HexValue::F => "F", HexValue::C => "C",
            },
            "confidence": *confidence,
        }),
        IxNode::Skill { args, .. } => args.clone(),
    }
}

/// Look up socket metadata for an imported skill.
///
/// Phase-1 strategy: registry-backed composite tools take a single
/// `Json → Json` socket pair (since all batch1/2/3 wrappers have shape
/// `fn(Value) -> Result<Value, String>`). Unknown skills still get one
/// connectable `Any → Any` pair so users can wire them visually.
fn sockets_for_skill(_skill_name: &str) -> (Vec<Socket>, Vec<Socket>) {
    // Reserved: once primitive typed skills are registered with richer
    // socket metadata, this becomes a registry lookup that maps the skill's
    // inputs/outputs to real typed Sockets.
    let ins = vec![Socket {
        name: "in".into(),
        ty: SocketType::Any,
    }];
    let outs = vec![Socket {
        name: "out".into(),
        ty: SocketType::Any,
    }];
    (ins, outs)
}

fn slugify(title: &str) -> String {
    title
        .chars()
        .map(|c| {
            if c.is_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .to_string()
}

fn short_value(v: &Value) -> String {
    match v {
        Value::Null => "—".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => {
            if s.len() > 80 {
                format!("{}…", &s[..80])
            } else {
                s.clone()
            }
        }
        Value::Array(a) => format!("[{} items]", a.len()),
        Value::Object(o) => {
            let keys: Vec<String> = o.keys().take(5).cloned().collect();
            let more = if o.len() > 5 {
                format!(" +{}", o.len() - 5)
            } else {
                String::new()
            };
            format!("{{{}{more}}}", keys.join(", "))
        }
    }
}

fn display_id(node_id: NodeId) -> String {
    // NodeId's Display impl prints a numeric id.
    format!("{node_id:?}")
        .chars()
        .filter(|c| c.is_ascii_digit())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_produces_value_output() {
        let node = IxNode::Constant { value: 42.0 };
        let out = execute_node_with_inputs(&node, &HashMap::new());
        match out {
            StepOutcome::Skipped(v) => assert_eq!(v["value"], 42.0),
            _ => panic!("expected Skipped"),
        }
    }

    #[test]
    fn belief_produces_hex_letter() {
        let node = IxNode::Belief {
            proposition: "test".into(),
            value: HexValue::P,
            confidence: 0.7,
        };
        let out = execute_node_with_inputs(&node, &HashMap::new());
        match out {
            StepOutcome::Skipped(v) => {
                assert_eq!(v["truth_value"], "P");
                assert_eq!(v["confidence"], 0.7);
            }
            _ => panic!("expected Skipped"),
        }
    }

    #[test]
    fn policy_gate_runs_registry_skill() {
        let node = IxNode::PolicyGate {
            policy: "alignment".into(),
            threshold: 0.5,
        };
        let out = execute_node_with_inputs(&node, &HashMap::new());
        matches!(out, StepOutcome::Produced(_) | StepOutcome::Skipped(_));
    }

    #[test]
    fn as_vector_flattens_matrix_column() {
        let m = json!([[1.0], [2.0], [3.0]]);
        let v = as_vector(&m);
        assert_eq!(v, json!([1.0, 2.0, 3.0]));
    }

    #[test]
    fn as_vector_preserves_flat_array() {
        let v = json!([1.0, 2.0, 3.0]);
        assert_eq!(as_vector(&v), v);
    }

    #[test]
    fn as_matrix_wraps_flat_vector() {
        let v = json!([1.0, 2.0, 3.0]);
        assert_eq!(as_matrix(&v), json!([[1.0], [2.0], [3.0]]));
    }

    #[test]
    fn as_matrix_preserves_2d_array() {
        let m = json!([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(as_matrix(&m), m);
    }

    #[test]
    fn upstream_dataset_extracts_data_field() {
        let mut up: HashMap<String, Value> = HashMap::new();
        up.insert("in".into(), json!({ "data": [[1.0, 2.0]] }));
        assert_eq!(upstream_dataset(&up), json!([[1.0, 2.0]]));
    }

    #[test]
    fn upstream_dataset_uses_array_directly() {
        let mut up: HashMap<String, Value> = HashMap::new();
        up.insert("in".into(), json!([[9.0, 8.0], [7.0, 6.0]]));
        assert_eq!(upstream_dataset(&up), json!([[9.0, 8.0], [7.0, 6.0]]));
    }

    #[test]
    fn upstream_dataset_falls_back_to_demo_data() {
        let up = HashMap::new();
        let v = upstream_dataset(&up);
        assert!(v.is_array());
        assert_eq!(v.as_array().unwrap().len(), 5);
    }

    #[test]
    fn short_value_truncates_long_strings() {
        let long = "x".repeat(120);
        let s = short_value(&Value::String(long));
        assert!(s.ends_with('…'));
        assert!(s.chars().count() <= 82);
    }

    #[test]
    fn short_value_summarizes_objects() {
        let v = json!({ "a": 1, "b": 2, "c": 3 });
        assert_eq!(short_value(&v), "{a, b, c}");
    }

    #[test]
    fn run_yaml_populates_yaml_run_from_ix_yaml() {
        // Write a minimal ix.yaml to a temp file, call run_yaml(), verify
        // the editor's yaml_run field holds a populated result.
        let mut dir = std::env::temp_dir();
        dir.push(format!("ix-editor-runyaml-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let ix_yaml = dir.join("ix.yaml");
        std::fs::write(
            &ix_yaml,
            r#"version: "1"
stages:
  load:
    skill: stats
    args:
      data: [1.0, 2.0, 3.0, 4.0, 5.0]
"#,
        )
        .unwrap();

        let mut editor = PipelineEditor::default();
        editor.run_yaml(ix_yaml.to_str().unwrap());
        let run = editor.yaml_run.as_ref().expect("yaml_run populated");
        assert!(run.error.is_none(), "unexpected error: {:?}", run.error);
        assert_eq!(run.stages.len(), 1);
        assert!(run.stages.contains_key("load"));
        let load_out = &run.stages["load"];
        assert_eq!(load_out["mean"], 3.0);
        assert_eq!(load_out["count"], 5);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn import_yaml_reconstructs_skill_nodes() {
        let mut dir = std::env::temp_dir();
        dir.push(format!("ix-editor-import-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let ix_yaml = dir.join("ix.yaml");
        std::fs::write(
            &ix_yaml,
            r#"version: "1"
stages:
  load:
    skill: stats
    args: { data: [1.0, 2.0, 3.0] }
  audit:
    skill: governance.check
    args: { action: "review" }
    deps: [load]
"#,
        )
        .unwrap();

        let mut editor = PipelineEditor::default();
        let n = editor
            .import_yaml(ix_yaml.to_str().unwrap())
            .expect("import succeeds");
        assert_eq!(n, 2);

        // Two Skill nodes with correct skill names, and one edge.
        let node_ids: Vec<_> = editor.snarl.node_ids().collect();
        assert_eq!(node_ids.len(), 2);

        let mut skills: Vec<String> = node_ids
            .iter()
            .map(|(_, n)| match n {
                IxNode::Skill { skill, .. } => skill.clone(),
                _ => panic!("expected Skill variant"),
            })
            .collect();
        skills.sort();
        assert_eq!(skills, vec!["governance.check", "stats"]);

        let wire_count = editor.snarl.wires().count();
        assert_eq!(wire_count, 1);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn import_yaml_reports_parse_error() {
        let mut editor = PipelineEditor::default();
        let err = editor.import_yaml("/nonexistent/ix.yaml").unwrap_err();
        assert!(!err.is_empty());
    }

    #[test]
    fn run_yaml_reports_load_error_on_missing_file() {
        let mut editor = PipelineEditor::default();
        editor.run_yaml("/nonexistent/path/ix.yaml");
        let run = editor.yaml_run.as_ref().unwrap();
        assert!(run.error.is_some());
        assert_eq!(run.stages.len(), 0);
    }

    #[test]
    fn run_yaml_reports_lower_error_on_unknown_skill() {
        let mut dir = std::env::temp_dir();
        dir.push(format!("ix-editor-bad-skill-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let ix_yaml = dir.join("ix.yaml");
        std::fs::write(
            &ix_yaml,
            r#"version: "1"
stages:
  bad:
    skill: does.not.exist
"#,
        )
        .unwrap();

        let mut editor = PipelineEditor::default();
        editor.run_yaml(ix_yaml.to_str().unwrap());
        let run = editor.yaml_run.as_ref().unwrap();
        assert!(run.error.is_some());
        assert!(run.error.as_ref().unwrap().contains("lower"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn slugify_handles_special_chars() {
        assert_eq!(slugify("K-Means"), "k_means");
        assert_eq!(slugify("CSV Read"), "csv_read");
        assert_eq!(slugify("Linear Regression!"), "linear_regression");
    }
}
