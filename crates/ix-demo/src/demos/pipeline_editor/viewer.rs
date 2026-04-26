//! `IxViewer` — the `SnarlViewer<IxNode>` impl that renders nodes, typed
//! socket pins, and validates connections.
//!
//! The viewer is constructed per-frame with borrowed references to editor
//! state (search filter + run status) so nothing lives on the viewer itself
//! between frames.

use super::nodes::{HexValue, IxNode, NormMethod};
use super::sockets::SocketType;
use super::RunStatus;
use eframe::egui;
use eframe::egui::{Color32, Stroke, Ui};
use egui_snarl::ui::{PinInfo, SnarlPin, SnarlViewer};
use egui_snarl::{InPin, NodeId, OutPin, Snarl};
use std::collections::HashMap;

pub struct IxViewer<'a> {
    pub search: &'a str,
    pub run_status: &'a HashMap<NodeId, RunStatus>,
}

impl<'a> IxViewer<'a> {
    fn menu_matches(&self, title: &str) -> bool {
        let q = self.search.trim().to_lowercase();
        q.is_empty() || title.to_lowercase().contains(&q)
    }
}

impl<'a> SnarlViewer<IxNode> for IxViewer<'a> {
    fn title(&mut self, node: &IxNode) -> String {
        node.title().to_string()
    }

    fn inputs(&mut self, node: &IxNode) -> usize {
        node.input_sockets().len()
    }

    fn outputs(&mut self, node: &IxNode) -> usize {
        node.output_sockets().len()
    }

    fn show_input(
        &mut self,
        pin: &InPin,
        ui: &mut Ui,
        _scale: f32,
        snarl: &mut Snarl<IxNode>,
    ) -> impl SnarlPin + 'static {
        let node = &snarl[pin.id.node];
        let sockets = node.input_sockets();
        let socket = &sockets[pin.id.input];
        ui.label(&socket.name);
        PinInfo::circle().with_fill(socket.ty.color())
    }

    fn show_output(
        &mut self,
        pin: &OutPin,
        ui: &mut Ui,
        _scale: f32,
        snarl: &mut Snarl<IxNode>,
    ) -> impl SnarlPin + 'static {
        let node = &snarl[pin.id.node];
        let sockets = node.output_sockets();
        let socket = &sockets[pin.id.output];
        ui.label(&socket.name);
        PinInfo::circle().with_fill(socket.ty.color())
    }

    fn connect(&mut self, from: &OutPin, to: &InPin, snarl: &mut Snarl<IxNode>) {
        let src_ty = snarl[from.id.node].output_sockets()[from.id.output].ty;
        let dst_ty = snarl[to.id.node].input_sockets()[to.id.input].ty;
        if !SocketType::compatible_with(src_ty, dst_ty) {
            return;
        }
        for existing in to.remotes.iter().copied() {
            snarl.disconnect(existing, to.id);
        }
        snarl.connect(from.id, to.id);
    }

    fn show_header(
        &mut self,
        node_id: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut Ui,
        _scale: f32,
        snarl: &mut Snarl<IxNode>,
    ) {
        let node = &snarl[node_id];
        let status = self.run_status.get(&node_id);

        // Status dot (8px) before the title.
        let dot_color = match status {
            Some(RunStatus::Ok) => Color32::from_rgb(0x22, 0xc5, 0x5e),
            Some(RunStatus::Err(_)) => Color32::from_rgb(0xef, 0x44, 0x44),
            Some(RunStatus::Skipped) => Color32::from_rgb(0xa3, 0xa3, 0xa3),
            None => Color32::TRANSPARENT,
        };

        ui.horizontal(|ui| {
            if status.is_some() {
                let (rect, resp) =
                    ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                ui.painter().circle_filled(rect.center(), 4.0, dot_color);
                if let Some(RunStatus::Err(msg)) = status {
                    resp.on_hover_text(msg);
                }
            }
            if node.is_gate() {
                let color = Color32::from_rgb(0xE0, 0x3A, 0x3A);
                ui.colored_label(color, format!("⚖ {}", node.title()));
            } else {
                ui.label(node.title());
            }
            // For Skill nodes, show the dotted skill name beside the label.
            if let IxNode::Skill { skill, .. } = node {
                ui.label(
                    egui::RichText::new(format!("({skill})"))
                        .weak()
                        .small()
                        .monospace(),
                );
            }
        });

        if node.is_gate() {
            // Red border around the whole header row.
            let color = Color32::from_rgb(0xE0, 0x3A, 0x3A);
            let r = ui.min_rect();
            ui.painter().rect_stroke(
                r.expand(1.0),
                4.0,
                Stroke::new(2.0, color),
                egui::StrokeKind::Outside,
            );
        }
    }

    fn has_body(&mut self, _node: &IxNode) -> bool {
        true
    }

    fn show_body(
        &mut self,
        node_id: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut Ui,
        _scale: f32,
        snarl: &mut Snarl<IxNode>,
    ) {
        match &mut snarl[node_id] {
            IxNode::CsvRead { path } | IxNode::CsvWrite { path } => {
                ui.horizontal(|ui| {
                    ui.label("path:");
                    ui.text_edit_singleline(path);
                });
            }
            IxNode::Constant { value } => {
                ui.add(egui::DragValue::new(value).speed(0.1));
            }
            IxNode::Normalize { method } => {
                egui::ComboBox::from_label("method")
                    .selected_text(format!("{method:?}"))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(method, NormMethod::ZScore, "ZScore");
                        ui.selectable_value(method, NormMethod::MinMax, "MinMax");
                        ui.selectable_value(method, NormMethod::L2, "L2");
                    });
            }
            IxNode::KMeans { k, max_iter, seed } => {
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(k).prefix("k=").range(1..=64));
                    ui.add(
                        egui::DragValue::new(max_iter)
                            .prefix("iter=")
                            .range(1..=1000),
                    );
                });
                ui.add(egui::DragValue::new(seed).prefix("seed="));
            }
            IxNode::Fft { inverse } => {
                ui.checkbox(inverse, "inverse");
            }
            IxNode::PolicyGate { policy, threshold } => {
                ui.horizontal(|ui| {
                    ui.label("policy:");
                    ui.text_edit_singleline(policy);
                });
                ui.add(egui::Slider::new(threshold, 0.0..=1.0).text("τ"));
            }
            IxNode::Plot { title } => {
                ui.horizontal(|ui| {
                    ui.label("title:");
                    ui.text_edit_singleline(title);
                });
            }
            IxNode::LinearReg => {
                ui.weak("(no params)");
            }
            IxNode::Belief {
                proposition,
                value,
                confidence,
            } => {
                ui.horizontal(|ui| {
                    ui.label("proposition:");
                    ui.text_edit_singleline(proposition);
                });
                // 2x3 hexavalent quadrant selector + confidence bar.
                draw_hex_badge(ui, value);
                ui.add(egui::Slider::new(confidence, 0.0..=1.0).text("confidence"));
            }
            IxNode::Skill { args, .. } => {
                // Show the JSON args as a read-only collapsed summary.
                let summary = match args {
                    serde_json::Value::Object(m) => format!("{{{} keys}}", m.len()),
                    serde_json::Value::Null => "—".into(),
                    other => {
                        let s = other.to_string();
                        if s.len() > 48 {
                            format!("{}…", &s[..48])
                        } else {
                            s
                        }
                    }
                };
                ui.label(
                    egui::RichText::new(format!("args: {summary}"))
                        .weak()
                        .small()
                        .monospace(),
                );
            }
        }
    }

    fn has_graph_menu(&mut self, _p: egui::Pos2, _s: &mut Snarl<IxNode>) -> bool {
        true
    }

    fn show_graph_menu(
        &mut self,
        pos: egui::Pos2,
        ui: &mut Ui,
        _scale: f32,
        snarl: &mut Snarl<IxNode>,
    ) {
        ui.label(egui::RichText::new("Add node").strong());
        ui.separator();
        if !self.search.trim().is_empty() {
            ui.label(format!("filter: {}", self.search));
        }

        let mut add = |ui: &mut Ui, label: &str, node: IxNode, matches: bool| {
            if matches && ui.button(label).clicked() {
                snarl.insert_node(pos, node);
                ui.close_menu();
            }
        };

        ui.label(egui::RichText::new("IO").strong().small());
        add(
            ui,
            "CSV Read",
            IxNode::CsvRead {
                path: String::new(),
            },
            self.menu_matches("CSV Read"),
        );
        add(
            ui,
            "CSV Write",
            IxNode::CsvWrite {
                path: String::new(),
            },
            self.menu_matches("CSV Write"),
        );
        add(
            ui,
            "Constant",
            IxNode::Constant { value: 0.0 },
            self.menu_matches("Constant"),
        );

        ui.label(egui::RichText::new("Transform").strong().small());
        add(
            ui,
            "Normalize",
            IxNode::Normalize {
                method: NormMethod::ZScore,
            },
            self.menu_matches("Normalize"),
        );
        add(
            ui,
            "FFT",
            IxNode::Fft { inverse: false },
            self.menu_matches("FFT"),
        );

        ui.label(egui::RichText::new("ML").strong().small());
        add(
            ui,
            "K-Means",
            IxNode::KMeans {
                k: 3,
                max_iter: 100,
                seed: 42,
            },
            self.menu_matches("K-Means"),
        );
        add(
            ui,
            "Linear Reg",
            IxNode::LinearReg,
            self.menu_matches("Linear Reg"),
        );

        ui.label(egui::RichText::new("Governance").strong().small());
        add(
            ui,
            "Policy Gate",
            IxNode::PolicyGate {
                policy: "alignment".into(),
                threshold: 0.7,
            },
            self.menu_matches("Policy Gate"),
        );
        add(
            ui,
            "Belief",
            IxNode::Belief {
                proposition: "API is stable".into(),
                value: HexValue::P,
                confidence: 0.7,
            },
            self.menu_matches("Belief"),
        );

        ui.label(egui::RichText::new("Sink").strong().small());
        add(
            ui,
            "Plot",
            IxNode::Plot {
                title: "plot".into(),
            },
            self.menu_matches("Plot"),
        );
    }

    fn has_node_menu(&mut self, _node: &IxNode) -> bool {
        true
    }

    fn show_node_menu(
        &mut self,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut Ui,
        _scale: f32,
        snarl: &mut Snarl<IxNode>,
    ) {
        if ui.button("Remove").clicked() {
            snarl.remove_node(node);
            ui.close_menu();
        }
    }
}

/// Draw the 2x3 hexavalent quadrant badge. User clicks a cell to select it.
fn draw_hex_badge(ui: &mut Ui, selected: &mut HexValue) {
    ui.horizontal(|ui| {
        for h in HexValue::all() {
            let is_selected = h == *selected;
            let color = h.color();
            let (rect, resp) = ui.allocate_exact_size(egui::vec2(24.0, 24.0), egui::Sense::click());
            let fill = if is_selected {
                color
            } else {
                color.gamma_multiply(0.35)
            };
            ui.painter().rect_filled(rect, 3.0, fill);
            if is_selected {
                ui.painter().rect_stroke(
                    rect,
                    3.0,
                    Stroke::new(1.5, Color32::WHITE),
                    egui::StrokeKind::Outside,
                );
            }
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                h.letter(),
                egui::FontId::monospace(14.0),
                Color32::WHITE,
            );
            if resp.clicked() {
                *selected = h;
            }
            let _ = is_selected; // consumed via styling above
            resp.on_hover_text(match h {
                HexValue::T => "True — verified",
                HexValue::P => "Probable — leans true",
                HexValue::U => "Unknown — no evidence",
                HexValue::D => "Doubtful — leans false",
                HexValue::F => "False — refuted",
                HexValue::C => "Contradictory — conflicting",
            });
        }
    });
}
