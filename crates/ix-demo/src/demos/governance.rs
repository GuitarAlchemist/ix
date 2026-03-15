use eframe::egui;
use ix_governance::{
    BeliefState, Constitution, EvidenceItem, Persona, TruthValue,
    list_personas,
};

fn governance_dir() -> std::path::PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    std::path::PathBuf::from(manifest).join("../../governance/demerzel")
}

pub struct GovernanceDemo {
    // Tetravalent logic
    tv_a: TruthValue,
    tv_b: TruthValue,

    // Constitution
    constitution: Option<Constitution>,
    constitution_error: Option<String>,

    // Belief state
    belief_proposition: String,
    belief_confidence: f64,
    belief_tv: TruthValue,
    belief_support_text: String,
    belief_contra_text: String,
    belief_state: Option<BeliefState>,

    // Personas
    persona_names: Vec<String>,
    persona_error: Option<String>,
    selected_persona_idx: usize,
    loaded_persona: Option<Persona>,
    persona_load_error: Option<String>,
}

impl Default for GovernanceDemo {
    fn default() -> Self {
        let constitution = Constitution::load(
            &governance_dir().join("constitutions/default.constitution.md"),
        ).ok();
        let constitution_error = if constitution.is_none() {
            Some("Failed to load constitution from governance/demerzel/constitutions/".into())
        } else {
            None
        };

        let (persona_names, persona_error) = match list_personas(&governance_dir().join("personas"))
        {
            Ok(names) => (names, None),
            Err(e) => (Vec::new(), Some(format!("Failed to list personas: {}", e))),
        };

        Self {
            tv_a: TruthValue::True,
            tv_b: TruthValue::False,

            constitution,
            constitution_error,

            belief_proposition: "The API is stable".into(),
            belief_confidence: 0.5,
            belief_tv: TruthValue::Unknown,
            belief_support_text: String::new(),
            belief_contra_text: String::new(),
            belief_state: None,

            persona_names,
            persona_error,
            selected_persona_idx: 0,
            loaded_persona: None,
            persona_load_error: None,
        }
    }
}

impl GovernanceDemo {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Governance (ix-governance)");
        ui.label("Demerzel agent governance: tetravalent logic, constitution, beliefs, and personas.");
        ui.add_space(8.0);

        self.tetravalent_section(ui);
        ui.add_space(12.0);
        self.constitution_section(ui);
        ui.add_space(12.0);
        self.belief_section(ui);
        ui.add_space(12.0);
        self.persona_section(ui);
    }

    fn tetravalent_section(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new(egui::RichText::new("Tetravalent Logic Explorer").strong())
            .default_open(true)
            .show(ui, |ui| {
                ui.label("Four-valued logic: True (T), False (F), Unknown (U), Contradictory (C)");
                ui.add_space(4.0);

                ui.horizontal(|ui| {
                    ui.label("A:");
                    tv_radio(ui, &mut self.tv_a, "a");
                    ui.add_space(16.0);
                    ui.label("B:");
                    tv_radio(ui, &mut self.tv_b, "b");
                });

                ui.add_space(4.0);

                egui::Grid::new("tv_results").striped(true).show(ui, |ui| {
                    ui.label("NOT A:");
                    ui.label(format!("{}", !self.tv_a));
                    ui.end_row();
                    ui.label("A AND B:");
                    ui.label(format!("{}", self.tv_a.and(self.tv_b)));
                    ui.end_row();
                    ui.label("A OR B:");
                    ui.label(format!("{}", self.tv_a.or(self.tv_b)));
                    ui.end_row();
                });

                ui.add_space(8.0);
                ui.label(egui::RichText::new("AND Truth Table").strong());
                truth_table_grid(ui, "and_table", |a, b| a.and(b));

                ui.add_space(8.0);
                ui.label(egui::RichText::new("OR Truth Table").strong());
                truth_table_grid(ui, "or_table", |a, b| a.or(b));
            });
    }

    fn constitution_section(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new(egui::RichText::new("Constitution Articles").strong())
            .default_open(false)
            .show(ui, |ui| {
                if let Some(err) = &self.constitution_error {
                    ui.colored_label(egui::Color32::RED, err);
                    return;
                }
                if let Some(c) = &self.constitution {
                    ui.label(format!("Version: {}", c.version));
                    ui.add_space(4.0);
                    for article in &c.articles {
                        egui::CollapsingHeader::new(format!(
                            "Article {}: {}",
                            article.number, article.name
                        ))
                        .id_salt(format!("article_{}", article.number))
                        .show(ui, |ui| {
                            ui.label(&article.text);
                        });
                    }
                }
            });
    }

    fn belief_section(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new(egui::RichText::new("Belief State Builder").strong())
            .default_open(false)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Proposition:");
                    ui.text_edit_singleline(&mut self.belief_proposition);
                });

                ui.horizontal(|ui| {
                    ui.label("Confidence:");
                    ui.add(egui::Slider::new(&mut self.belief_confidence, 0.0..=1.0));
                });

                ui.horizontal(|ui| {
                    ui.label("Initial truth value:");
                    tv_radio(ui, &mut self.belief_tv, "belief");
                });

                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label("Supporting evidence:");
                    ui.text_edit_singleline(&mut self.belief_support_text);
                    if ui.button("Add").clicked() && !self.belief_support_text.is_empty() {
                        let text = self.belief_support_text.clone();
                        self.belief_support_text.clear();
                        let bs = self.belief_state.get_or_insert_with(|| {
                            BeliefState::new(
                                self.belief_proposition.clone(),
                                self.belief_tv,
                                self.belief_confidence,
                            )
                        });
                        bs.add_supporting(EvidenceItem {
                            source: "user".into(),
                            claim: text,
                        });
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Contradicting evidence:");
                    ui.text_edit_singleline(&mut self.belief_contra_text);
                    if ui.button("Add").clicked() && !self.belief_contra_text.is_empty() {
                        let text = self.belief_contra_text.clone();
                        self.belief_contra_text.clear();
                        let bs = self.belief_state.get_or_insert_with(|| {
                            BeliefState::new(
                                self.belief_proposition.clone(),
                                self.belief_tv,
                                self.belief_confidence,
                            )
                        });
                        bs.add_contradicting(EvidenceItem {
                            source: "user".into(),
                            claim: text,
                        });
                    }
                });

                if ui.button("Create / Reset Belief").clicked() {
                    self.belief_state = Some(BeliefState::new(
                        self.belief_proposition.clone(),
                        self.belief_tv,
                        self.belief_confidence,
                    ));
                }

                if let Some(bs) = &self.belief_state {
                    ui.add_space(8.0);
                    ui.separator();
                    egui::Grid::new("belief_grid").striped(true).show(ui, |ui| {
                        ui.label("Proposition:");
                        ui.label(&bs.proposition);
                        ui.end_row();
                        ui.label("Truth value:");
                        ui.label(format!("{}", bs.truth_value));
                        ui.end_row();
                        ui.label("Confidence:");
                        ui.label(format!("{:.2}", bs.confidence));
                        ui.end_row();
                        ui.label("Supporting:");
                        ui.label(format!("{} item(s)", bs.supporting.len()));
                        ui.end_row();
                        ui.label("Contradicting:");
                        ui.label(format!("{} item(s)", bs.contradicting.len()));
                        ui.end_row();
                        ui.label("Resolved action:");
                        ui.label(format!("{:?}", bs.resolve()));
                        ui.end_row();
                    });

                    if !bs.supporting.is_empty() || !bs.contradicting.is_empty() {
                        ui.add_space(4.0);
                        for e in &bs.supporting {
                            ui.label(format!("  [+] {}: {}", e.source, e.claim));
                        }
                        for e in &bs.contradicting {
                            ui.label(format!("  [-] {}: {}", e.source, e.claim));
                        }
                    }
                }
            });
    }

    fn persona_section(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new(egui::RichText::new("Persona Browser").strong())
            .default_open(false)
            .show(ui, |ui| {
                if let Some(err) = &self.persona_error {
                    ui.colored_label(egui::Color32::RED, err);
                    return;
                }

                if self.persona_names.is_empty() {
                    ui.label("No personas found.");
                    return;
                }

                let prev_idx = self.selected_persona_idx;
                egui::ComboBox::from_label("Persona")
                    .selected_text(
                        self.persona_names
                            .get(self.selected_persona_idx)
                            .cloned()
                            .unwrap_or_default(),
                    )
                    .show_ui(ui, |ui| {
                        for (i, name) in self.persona_names.iter().enumerate() {
                            ui.selectable_value(&mut self.selected_persona_idx, i, name);
                        }
                    });

                // Load on change or first time
                if self.loaded_persona.is_none() || prev_idx != self.selected_persona_idx {
                    if let Some(name) = self.persona_names.get(self.selected_persona_idx) {
                        match Persona::load_by_name(
                            &governance_dir().join("personas"),
                            name,
                        ) {
                            Ok(p) => {
                                self.loaded_persona = Some(p);
                                self.persona_load_error = None;
                            }
                            Err(e) => {
                                self.loaded_persona = None;
                                self.persona_load_error =
                                    Some(format!("Failed to load persona: {}", e));
                            }
                        }
                    }
                }

                if let Some(err) = &self.persona_load_error {
                    ui.colored_label(egui::Color32::RED, err);
                }

                if let Some(p) = &self.loaded_persona {
                    ui.add_space(4.0);
                    egui::Grid::new("persona_grid").striped(true).show(ui, |ui| {
                        ui.label("Name:");
                        ui.label(&p.name);
                        ui.end_row();
                        ui.label("Role:");
                        ui.label(&p.role);
                        ui.end_row();
                        ui.label("Domain:");
                        ui.label(&p.domain);
                        ui.end_row();
                        ui.label("Voice tone:");
                        ui.label(&p.voice.tone);
                        ui.end_row();
                        ui.label("Voice verbosity:");
                        ui.label(&p.voice.verbosity);
                        ui.end_row();
                        ui.label("Voice style:");
                        ui.label(&p.voice.style);
                        ui.end_row();
                    });

                    if !p.capabilities.is_empty() {
                        ui.add_space(4.0);
                        ui.label(egui::RichText::new("Capabilities:").strong());
                        for cap in &p.capabilities {
                            ui.label(format!("  - {}", cap));
                        }
                    }

                    if !p.constraints.is_empty() {
                        ui.add_space(4.0);
                        ui.label(egui::RichText::new("Constraints:").strong());
                        for con in &p.constraints {
                            ui.label(format!("  - {}", con));
                        }
                    }
                }
            });
    }
}

fn tv_radio(ui: &mut egui::Ui, val: &mut TruthValue, id_prefix: &str) {
    let _ = id_prefix; // used for uniqueness via the surrounding context
    ui.radio_value(val, TruthValue::True, "T");
    ui.radio_value(val, TruthValue::False, "F");
    ui.radio_value(val, TruthValue::Unknown, "U");
    ui.radio_value(val, TruthValue::Contradictory, "C");
}

fn truth_table_grid(ui: &mut egui::Ui, id: &str, op: fn(TruthValue, TruthValue) -> TruthValue) {
    let values = [
        TruthValue::True,
        TruthValue::False,
        TruthValue::Unknown,
        TruthValue::Contradictory,
    ];
    let labels = ["T", "F", "U", "C"];

    egui::Grid::new(id).striped(true).show(ui, |ui| {
        // Header row
        ui.label("");
        for lbl in &labels {
            ui.label(egui::RichText::new(*lbl).strong());
        }
        ui.end_row();

        for (i, &a) in values.iter().enumerate() {
            ui.label(egui::RichText::new(labels[i]).strong());
            for &b in &values {
                let result = op(a, b);
                let color = match result {
                    TruthValue::True => egui::Color32::from_rgb(80, 180, 80),
                    TruthValue::False => egui::Color32::from_rgb(200, 80, 80),
                    TruthValue::Unknown => egui::Color32::from_rgb(180, 180, 80),
                    TruthValue::Contradictory => egui::Color32::from_rgb(180, 80, 180),
                };
                ui.label(egui::RichText::new(format!("{}", result)).color(color));
            }
            ui.end_row();
        }
    });
}
