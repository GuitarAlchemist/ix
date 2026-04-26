use eframe::egui;

#[derive(Debug, Clone, Copy, PartialEq)]
enum CategoryMode {
    Monad,
    FreeForgetful,
    MonadLaws,
}

pub struct CategoryDemo {
    mode: CategoryMode,
    // Monad demo
    monad_input: String,
    monad_type: MonadType,
    monad_result: Option<MonadResult>,
    // Free-Forgetful adjunction
    adj_input: String,
    adj_result: Option<AdjResult>,
    // Monad laws verifier
    laws_input: String,
    laws_result: Option<LawsResult>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum MonadType {
    OptionMonad,
    ResultMonad,
}

struct MonadResult {
    initial: String,
    after_unit: String,
    after_safe_div: String,
    after_add_one: String,
}

struct AdjResult {
    input: Vec<i32>,
    after_free: Vec<Vec<i32>>,
    after_forget: Vec<i32>,
    round_trip_ok: bool,
}

struct LawsResult {
    value: i32,
    left_unit_pass: bool,
    left_unit_detail: String,
    right_unit_pass: bool,
    right_unit_detail: String,
    associativity_pass: bool,
    associativity_detail: String,
}

impl Default for CategoryDemo {
    fn default() -> Self {
        Self {
            mode: CategoryMode::Monad,
            monad_input: "10".to_string(),
            monad_type: MonadType::OptionMonad,
            monad_result: None,
            adj_input: "1, 2, 3, 4".to_string(),
            adj_result: None,
            laws_input: "5".to_string(),
            laws_result: None,
        }
    }
}

impl CategoryDemo {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Category Theory (ix-category)");

        ui.horizontal(|ui| {
            ui.radio_value(&mut self.mode, CategoryMode::Monad, "Monad Demo");
            ui.radio_value(
                &mut self.mode,
                CategoryMode::FreeForgetful,
                "Free-Forgetful Adjunction",
            );
            ui.radio_value(
                &mut self.mode,
                CategoryMode::MonadLaws,
                "Monad Laws Verifier",
            );
        });

        ui.separator();

        match self.mode {
            CategoryMode::Monad => self.ui_monad(ui),
            CategoryMode::FreeForgetful => self.ui_free_forgetful(ui),
            CategoryMode::MonadLaws => self.ui_monad_laws(ui),
        }
    }

    fn ui_monad(&mut self, ui: &mut egui::Ui) {
        ui.label(
            egui::RichText::new("Monad: unit, bind, chain")
                .strong()
                .size(16.0),
        );
        ui.label("Input an integer, then chain operations: safe_div(100/x) and add_one(x+1).");

        ui.horizontal(|ui| {
            ui.label("Monad type:");
            ui.radio_value(&mut self.monad_type, MonadType::OptionMonad, "Option");
            ui.radio_value(&mut self.monad_type, MonadType::ResultMonad, "Result");
        });

        ui.horizontal(|ui| {
            ui.label("Input value:");
            ui.text_edit_singleline(&mut self.monad_input);
        });

        if ui.button("Run Monad Chain").clicked() {
            self.run_monad();
        }

        if let Some(r) = &self.monad_result {
            egui::Grid::new("monad_grid").striped(true).show(ui, |ui| {
                ui.label("Step");
                ui.label("Value");
                ui.end_row();
                ui.label("Parse input:");
                ui.label(&r.initial);
                ui.end_row();
                ui.label("unit(x):");
                ui.label(&r.after_unit);
                ui.end_row();
                ui.label("bind(safe_div = 100/x):");
                ui.label(&r.after_safe_div);
                ui.end_row();
                ui.label("bind(add_one = x+1):");
                ui.label(&r.after_add_one);
                ui.end_row();
            });
        }
    }

    fn run_monad(&mut self) {
        let parsed = self.monad_input.trim().parse::<i32>();

        match self.monad_type {
            MonadType::OptionMonad => {
                use ix_category::monad::{Monad, OptionMonad};

                let initial_str = match &parsed {
                    Ok(v) => format!("Some({})", v),
                    Err(e) => format!("None (parse error: {})", e),
                };

                let wrapped = parsed.ok();
                let after_unit = match wrapped {
                    Some(v) => {
                        let u = OptionMonad::unit(v);
                        format!("{:?}", u)
                    }
                    None => "None (skipped)".to_string(),
                };

                let safe_div = |x: i32| -> Option<i32> {
                    if x == 0 {
                        None
                    } else {
                        Some(100 / x)
                    }
                };
                let after_safe_div_val: Option<i32> = match wrapped {
                    Some(v) => OptionMonad::bind(OptionMonad::unit(v), safe_div),
                    None => None,
                };
                let after_safe_div = format!("{:?}", after_safe_div_val);

                let add_one = |x: i32| -> Option<i32> { Some(x + 1) };
                let after_add_one_val: Option<i32> = match after_safe_div_val {
                    Some(v) => OptionMonad::bind(Some(v), add_one),
                    None => None,
                };
                let after_add_one = format!("{:?}", after_add_one_val);

                self.monad_result = Some(MonadResult {
                    initial: initial_str,
                    after_unit,
                    after_safe_div,
                    after_add_one,
                });
            }
            MonadType::ResultMonad => {
                use ix_category::monad::{Monad, ResultMonad};

                let initial_str = match &parsed {
                    Ok(v) => format!("Ok({})", v),
                    Err(e) => format!("Err(\"{}\")", e),
                };

                let wrapped: Result<i32, String> = parsed.map_err(|e| e.to_string());
                let after_unit = match &wrapped {
                    Ok(v) => {
                        let u = ResultMonad::unit(*v);
                        format!("{:?}", u)
                    }
                    Err(e) => format!("Err(\"{}\") (skipped)", e),
                };

                let safe_div = |x: i32| -> Result<i32, String> {
                    if x == 0 {
                        Err("division by zero".to_string())
                    } else {
                        Ok(100 / x)
                    }
                };
                let after_safe_div_val: Result<i32, String> = match &wrapped {
                    Ok(v) => ResultMonad::bind(ResultMonad::unit(*v), safe_div),
                    Err(e) => Err(e.clone()),
                };
                let after_safe_div = format!("{:?}", after_safe_div_val);

                let add_one = |x: i32| -> Result<i32, String> { Ok(x + 1) };
                let after_add_one_val: Result<i32, String> = match &after_safe_div_val {
                    Ok(v) => ResultMonad::bind(Ok(*v), add_one),
                    Err(e) => Err(e.clone()),
                };
                let after_add_one = format!("{:?}", after_add_one_val);

                self.monad_result = Some(MonadResult {
                    initial: initial_str,
                    after_unit,
                    after_safe_div,
                    after_add_one,
                });
            }
        }
    }

    fn ui_free_forgetful(&mut self, ui: &mut egui::Ui) {
        ui.label(
            egui::RichText::new("Free ⊣ Forgetful Adjunction")
                .strong()
                .size(16.0),
        );
        ui.label(
            "Free functor wraps each element in a singleton list. Forgetful functor flattens back.",
        );
        ui.label("The round-trip Forget(Free(S)) should equal the original set S.");

        ui.horizontal(|ui| {
            ui.label("Integers (comma-separated):");
            ui.text_edit_singleline(&mut self.adj_input);
        });

        if ui.button("Apply Free → Forgetful").clicked() {
            self.run_adjunction();
        }

        if let Some(r) = &self.adj_result {
            egui::Grid::new("adj_grid").striped(true).show(ui, |ui| {
                ui.label("Input S:");
                ui.label(format!("{:?}", r.input));
                ui.end_row();

                ui.label("Free(S) = List(S):");
                ui.label(format!("{:?}", r.after_free));
                ui.end_row();

                ui.label("Forget(Free(S)):");
                ui.label(format!("{:?}", r.after_forget));
                ui.end_row();

                ui.label("Round-trip S == Forget(Free(S)):");
                let status = if r.round_trip_ok { "PASS" } else { "FAIL" };
                let color = if r.round_trip_ok {
                    egui::Color32::GREEN
                } else {
                    egui::Color32::RED
                };
                ui.label(egui::RichText::new(status).color(color).strong());
                ui.end_row();
            });
        }
    }

    fn run_adjunction(&mut self) {
        use ix_category::monad::FreeForgetfulAdj;

        let input: Vec<i32> = self
            .adj_input
            .split(',')
            .filter_map(|s| s.trim().parse::<i32>().ok())
            .collect();

        let after_free = FreeForgetfulAdj::free(&input);
        let after_forget = FreeForgetfulAdj::forget(&after_free);
        let round_trip_ok = input == after_forget;

        self.adj_result = Some(AdjResult {
            input,
            after_free,
            after_forget,
            round_trip_ok,
        });
    }

    fn ui_monad_laws(&mut self, ui: &mut egui::Ui) {
        ui.label(
            egui::RichText::new("Monad Laws Verifier (OptionMonad)")
                .strong()
                .size(16.0),
        );
        ui.label("Verifies the three monad laws for OptionMonad with f(x) = Some(x + 1).");

        ui.horizontal(|ui| {
            ui.label("Value a:");
            ui.text_edit_singleline(&mut self.laws_input);
        });

        if ui.button("Verify Laws").clicked() {
            self.run_laws();
        }

        if let Some(r) = &self.laws_result {
            ui.label(format!("Testing with a = {}, f(x) = Some(x + 1)", r.value));
            ui.add_space(4.0);

            egui::Grid::new("laws_grid").striped(true).show(ui, |ui| {
                let pass_color = egui::Color32::GREEN;
                let fail_color = egui::Color32::RED;

                ui.label("Left unit: bind(unit(a), f) == f(a)");
                let (status, color) = if r.left_unit_pass {
                    ("PASS", pass_color)
                } else {
                    ("FAIL", fail_color)
                };
                ui.label(egui::RichText::new(status).color(color).strong());
                ui.label(&r.left_unit_detail);
                ui.end_row();

                ui.label("Right unit: bind(m, unit) == m");
                let (status, color) = if r.right_unit_pass {
                    ("PASS", pass_color)
                } else {
                    ("FAIL", fail_color)
                };
                ui.label(egui::RichText::new(status).color(color).strong());
                ui.label(&r.right_unit_detail);
                ui.end_row();

                ui.label("Associativity: bind(bind(m, f), g) == bind(m, |x| bind(f(x), g))");
                let (status, color) = if r.associativity_pass {
                    ("PASS", pass_color)
                } else {
                    ("FAIL", fail_color)
                };
                ui.label(egui::RichText::new(status).color(color).strong());
                ui.label(&r.associativity_detail);
                ui.end_row();
            });
        }
    }

    fn run_laws(&mut self) {
        use ix_category::monad::{Monad, OptionMonad};

        let a = match self.laws_input.trim().parse::<i32>() {
            Ok(v) => v,
            Err(_) => return,
        };

        let f = |x: i32| -> Option<i32> { Some(x + 1) };
        let g = |x: i32| -> Option<i32> { Some(x * 2) };

        // Left unit law: bind(unit(a), f) == f(a)
        let lhs_left: Option<i32> = OptionMonad::bind(OptionMonad::unit(a), f);
        let rhs_left = f(a);
        let left_unit_pass = lhs_left == rhs_left;
        let left_unit_detail = format!("{:?} == {:?}", lhs_left, rhs_left);

        // Right unit law: bind(m, unit) == m
        let m = OptionMonad::unit(a);
        let lhs_right: Option<i32> = OptionMonad::bind(m, OptionMonad::unit);
        let rhs_right = m;
        let right_unit_pass = lhs_right == rhs_right;
        let right_unit_detail = format!("{:?} == {:?}", lhs_right, rhs_right);

        // Associativity: bind(bind(m, f), g) == bind(m, |x| bind(f(x), g))
        let bind_m_f: Option<i32> = OptionMonad::bind(m, f);
        let lhs_assoc: Option<i32> = match bind_m_f {
            Some(v) => OptionMonad::bind(Some(v), g),
            None => None,
        };
        let rhs_assoc: Option<i32> = OptionMonad::bind(m, |x| {
            let fx = f(x);
            match fx {
                Some(v) => OptionMonad::bind(Some(v), g),
                None => None,
            }
        });
        let associativity_pass = lhs_assoc == rhs_assoc;
        let associativity_detail = format!("{:?} == {:?}", lhs_assoc, rhs_assoc);

        self.laws_result = Some(LawsResult {
            value: a,
            left_unit_pass,
            left_unit_detail,
            right_unit_pass,
            right_unit_detail,
            associativity_pass,
            associativity_detail,
        });
    }
}
