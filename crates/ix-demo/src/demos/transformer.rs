use eframe::egui;
use egui_plot::{Bar, BarChart, Line, Plot, PlotPoints, Points};
use ndarray::{s, Array2, Array3};

#[derive(PartialEq, Clone, Copy)]
enum TransformerMode {
    Attention,
    Positional,
    FullBlock,
}

pub struct TransformerDemo {
    mode: TransformerMode,
    // Attention
    seq_len: usize,
    d_model: usize,
    n_heads: usize,
    use_causal: bool,
    attn_weights: Option<Array2<f64>>,
    // Positional
    pe_max_len: usize,
    pe_d_model: usize,
    pe_data: Vec<Vec<f64>>,
    pe_type: PeType,
    // Full block
    n_layers: usize,
    d_ff: usize,
    input_data: Vec<[f64; 2]>,
    output_data: Vec<[f64; 2]>,
    status: String,
}

#[derive(PartialEq, Clone, Copy)]
enum PeType {
    Sinusoidal,
    Rope,
    Alibi,
}

impl Default for TransformerDemo {
    fn default() -> Self {
        Self {
            mode: TransformerMode::Attention,
            seq_len: 8,
            d_model: 16,
            n_heads: 4,
            use_causal: true,
            attn_weights: None,
            pe_max_len: 50,
            pe_d_model: 32,
            pe_data: Vec::new(),
            pe_type: PeType::Sinusoidal,
            n_layers: 2,
            d_ff: 64,
            input_data: Vec::new(),
            output_data: Vec::new(),
            status: "Ready".into(),
        }
    }
}

impl TransformerDemo {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Transformer (ix-nn)");
        ui.label("Pure-Rust transformer building blocks — no pretrained weights.");

        ui.horizontal(|ui| {
            ui.radio_value(&mut self.mode, TransformerMode::Attention, "Attention");
            ui.radio_value(
                &mut self.mode,
                TransformerMode::Positional,
                "Positional Encoding",
            );
            ui.radio_value(
                &mut self.mode,
                TransformerMode::FullBlock,
                "Transformer Stack",
            );
        });

        ui.separator();

        match self.mode {
            TransformerMode::Attention => self.attention_ui(ui),
            TransformerMode::Positional => self.positional_ui(ui),
            TransformerMode::FullBlock => self.full_block_ui(ui),
        }
    }

    fn attention_ui(&mut self, ui: &mut egui::Ui) {
        ui.label("Scaled dot-product attention — visualize attention weights.");
        ui.horizontal(|ui| {
            ui.label("Seq len:");
            ui.add(egui::Slider::new(&mut self.seq_len, 2..=16));
            ui.label("d_model:");
            ui.add(egui::Slider::new(&mut self.d_model, 4..=64));
            ui.label("Heads:");
            ui.add(egui::Slider::new(&mut self.n_heads, 1..=8));
        });
        ui.checkbox(&mut self.use_causal, "Causal mask (autoregressive)");

        if ui.button("Compute Attention").clicked() {
            self.run_attention();
        }

        ui.label(&self.status);

        // Draw attention heatmap as colored grid
        if let Some(ref w) = self.attn_weights {
            ui.label("Attention weights (head 0):");
            let n = w.nrows();
            let cell_size = (400.0 / n as f32).min(40.0);

            egui::Grid::new("attn_heatmap")
                .spacing([1.0, 1.0])
                .show(ui, |ui| {
                    // Header
                    ui.label("");
                    for j in 0..n {
                        ui.label(egui::RichText::new(format!("K{j}")).small());
                    }
                    ui.end_row();

                    for i in 0..n {
                        ui.label(egui::RichText::new(format!("Q{i}")).small());
                        for j in 0..n {
                            let val = w[[i, j]];
                            let intensity = (val * 255.0).clamp(0.0, 255.0) as u8;
                            let color =
                                egui::Color32::from_rgb(intensity, intensity / 3, 255 - intensity);
                            let (rect, _) = ui.allocate_exact_size(
                                egui::vec2(cell_size, cell_size),
                                egui::Sense::hover(),
                            );
                            ui.painter().rect_filled(rect, 0.0, color);
                            // Show value on hover
                            if ui.rect_contains_pointer(rect) {
                                egui::show_tooltip(
                                    ui.ctx(),
                                    ui.layer_id(),
                                    egui::Id::new(("attn_tip", i, j)),
                                    |ui| {
                                        ui.label(format!("Q{i}→K{j}: {val:.4}"));
                                    },
                                );
                            }
                        }
                        ui.end_row();
                    }
                });

            // Also show as bar chart for first query
            ui.label("Attention distribution for Q0:");
            let bars: Vec<Bar> = (0..n).map(|j| Bar::new(j as f64, w[[0, j]])).collect();
            Plot::new("attn_bars").height(150.0).show(ui, |plot_ui| {
                plot_ui.bar_chart(BarChart::new(bars).name("Q0 attention").width(0.7));
            });
        }
    }

    fn run_attention(&mut self) {
        use ix_nn::attention::{causal_mask, multi_head_attention};
        use rand::Rng;
        let mut rng = rand::rng();

        let mask = if self.use_causal {
            Some(causal_mask(self.seq_len))
        } else {
            None
        };

        // Generate random Q, K, V
        let q = Array3::from_shape_fn((1, self.seq_len, self.d_model), |_| {
            rng.random_range(-1.0..1.0)
        });
        let k = Array3::from_shape_fn((1, self.seq_len, self.d_model), |_| {
            rng.random_range(-1.0..1.0)
        });
        let v = Array3::from_shape_fn((1, self.seq_len, self.d_model), |_| {
            rng.random_range(-1.0..1.0)
        });

        // Use multi-head attention with identity projections to show raw attention
        let eye = Array2::from_diag(&ndarray::Array1::ones(self.d_model));

        let (_output, head_weights) = multi_head_attention(
            &q,
            &k,
            &v,
            &eye,
            &eye,
            &eye,
            &eye,
            self.n_heads,
            mask.as_ref(),
        );

        // Show first head's weights
        if let Some(hw) = head_weights.first() {
            self.attn_weights = Some(hw.slice(s![0, .., ..]).to_owned());
        }

        self.status = format!(
            "Computed {}-head attention: seq_len={}, d_model={}, d_k={}",
            self.n_heads,
            self.seq_len,
            self.d_model,
            self.d_model / self.n_heads
        );
    }

    fn positional_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.radio_value(&mut self.pe_type, PeType::Sinusoidal, "Sinusoidal");
            ui.radio_value(&mut self.pe_type, PeType::Rope, "RoPE effect");
            ui.radio_value(&mut self.pe_type, PeType::Alibi, "ALiBi bias");
        });
        ui.horizontal(|ui| {
            ui.label("Max len:");
            ui.add(egui::Slider::new(&mut self.pe_max_len, 10..=200));
            ui.label("d_model:");
            ui.add(egui::Slider::new(&mut self.pe_d_model, 4..=128));
        });

        if ui.button("Generate").clicked() {
            self.run_positional();
        }

        ui.label(&self.status);

        if !self.pe_data.is_empty() {
            match self.pe_type {
                PeType::Sinusoidal => {
                    ui.label("First 4 dimensions across positions:");
                    Plot::new("pe_plot").height(300.0).show(ui, |plot_ui| {
                        let colors = [
                            egui::Color32::from_rgb(255, 100, 100),
                            egui::Color32::from_rgb(100, 200, 100),
                            egui::Color32::from_rgb(100, 100, 255),
                            egui::Color32::from_rgb(255, 200, 50),
                        ];
                        for dim in 0..4.min(self.pe_data[0].len()) {
                            let pts: PlotPoints = self
                                .pe_data
                                .iter()
                                .enumerate()
                                .map(|(pos, row)| [pos as f64, row[dim]])
                                .collect();
                            plot_ui.line(
                                Line::new(pts)
                                    .name(format!("dim {dim}"))
                                    .width(2.0)
                                    .color(colors[dim % colors.len()]),
                            );
                        }
                    });
                }
                PeType::Rope => {
                    ui.label(
                        "Dot product between position 0 and position p (shows relative decay):",
                    );
                    Plot::new("rope_plot").height(300.0).show(ui, |plot_ui| {
                        let pts: PlotPoints = self.pe_data[0]
                            .iter()
                            .enumerate()
                            .map(|(i, &v)| [i as f64, v])
                            .collect();
                        plot_ui.line(Line::new(pts).name("dot(pos_0, pos_p)").width(2.0));
                    });
                }
                PeType::Alibi => {
                    ui.label("ALiBi bias matrix (4 heads):");
                    let n_show = self.pe_data.len().min(4);
                    Plot::new("alibi_plot").height(300.0).show(ui, |plot_ui| {
                        let colors = [
                            egui::Color32::from_rgb(255, 100, 100),
                            egui::Color32::from_rgb(100, 200, 100),
                            egui::Color32::from_rgb(100, 100, 255),
                            egui::Color32::from_rgb(255, 200, 50),
                        ];
                        for h in 0..n_show {
                            let pts: PlotPoints = self.pe_data[h]
                                .iter()
                                .enumerate()
                                .map(|(j, &v)| [j as f64, v])
                                .collect();
                            plot_ui.line(
                                Line::new(pts)
                                    .name(format!("head {h}"))
                                    .width(2.0)
                                    .color(colors[h % colors.len()]),
                            );
                        }
                    });
                }
            }
        }
    }

    fn run_positional(&mut self) {
        match self.pe_type {
            PeType::Sinusoidal => {
                let pe = ix_nn::positional::sinusoidal_encoding(self.pe_max_len, self.pe_d_model);
                self.pe_data = pe.rows().into_iter().map(|row| row.to_vec()).collect();
                self.status = format!(
                    "Sinusoidal PE: {} positions × {} dims",
                    self.pe_max_len, self.pe_d_model
                );
            }
            PeType::Rope => {
                // Show how dot product decays with position distance
                let d = self.pe_d_model;
                let n = self.pe_max_len;

                // Create vectors at each position and rotate them
                let base_vec =
                    Array2::from_shape_fn((n, d), |(_, j)| if j == 0 { 1.0 } else { 0.0 });
                let rotated = ix_nn::positional::rope_rotate(&base_vec, 10000.0);

                // Dot product of position 0 with all others
                let ref_row = rotated.row(0).to_owned();
                let dots: Vec<f64> = (0..n)
                    .map(|p| {
                        ref_row
                            .iter()
                            .zip(rotated.row(p).iter())
                            .map(|(a, b)| a * b)
                            .sum()
                    })
                    .collect();

                self.pe_data = vec![dots];
                self.status = format!("RoPE relative position encoding: {} positions, d={}", n, d);
            }
            PeType::Alibi => {
                let n_heads = 4;
                let slopes = ix_nn::positional::alibi_slopes(n_heads);
                let seq = self.pe_max_len.min(32); // limit for readability

                // For each head, show the last row of the bias matrix (how token at end attends)
                self.pe_data = slopes
                    .iter()
                    .map(|&slope| {
                        let bias = ix_nn::positional::alibi_bias(seq, slope);
                        let last_row = seq - 1;
                        (0..seq).map(|j| bias[[last_row, j]].max(-10.0)).collect()
                    })
                    .collect();

                self.status = format!(
                    "ALiBi: {} heads, slopes: [{}]",
                    n_heads,
                    slopes
                        .iter()
                        .map(|s| format!("{s:.4}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
        }
    }

    fn full_block_ui(&mut self, ui: &mut egui::Ui) {
        ui.label("Full transformer stack — pass random input through N layers.");
        ui.horizontal(|ui| {
            ui.label("Layers:");
            ui.add(egui::Slider::new(&mut self.n_layers, 1..=8));
            ui.label("d_model:");
            ui.add(egui::Slider::new(&mut self.d_model, 4..=64));
            ui.label("Heads:");
            ui.add(egui::Slider::new(&mut self.n_heads, 1..=8));
        });
        ui.horizontal(|ui| {
            ui.label("d_ff:");
            ui.add(egui::Slider::new(&mut self.d_ff, 8..=256));
            ui.label("Seq len:");
            ui.add(egui::Slider::new(&mut self.seq_len, 2..=16));
            ui.checkbox(&mut self.use_causal, "Causal mask");
        });

        if ui.button("Run Stack").clicked() {
            self.run_full_block();
        }

        ui.label(&self.status);

        if !self.input_data.is_empty() {
            ui.label("Input vs Output (first 2 dimensions, PCA-like projection):");
            Plot::new("transformer_io")
                .height(400.0)
                .data_aspect(1.0)
                .show(ui, |plot_ui| {
                    let in_pts: PlotPoints = self.input_data.iter().copied().collect();
                    let out_pts: PlotPoints = self.output_data.iter().copied().collect();

                    plot_ui.points(
                        Points::new(in_pts)
                            .radius(6.0)
                            .color(egui::Color32::from_rgb(100, 100, 255))
                            .name("Input tokens"),
                    );
                    plot_ui.points(
                        Points::new(out_pts)
                            .radius(6.0)
                            .color(egui::Color32::from_rgb(255, 100, 100))
                            .name("Output tokens"),
                    );

                    // Draw arrows from input to output
                    for (inp, outp) in self.input_data.iter().zip(self.output_data.iter()) {
                        let arrow: PlotPoints = vec![*inp, *outp].into_iter().collect();
                        plot_ui.line(
                            Line::new(arrow)
                                .width(1.0)
                                .color(egui::Color32::from_rgba_premultiplied(200, 200, 200, 100)),
                        );
                    }
                });

            // Norm comparison
            ui.label("Per-token L2 norm change:");
            let norms: Vec<Bar> = self
                .input_data
                .iter()
                .zip(self.output_data.iter())
                .enumerate()
                .map(|(i, (inp, outp))| {
                    let in_norm = (inp[0] * inp[0] + inp[1] * inp[1]).sqrt();
                    let out_norm = (outp[0] * outp[0] + outp[1] * outp[1]).sqrt();
                    Bar::new(i as f64, out_norm - in_norm)
                })
                .collect();
            Plot::new("norm_change").height(150.0).show(ui, |plot_ui| {
                plot_ui.bar_chart(BarChart::new(norms).name("Norm delta").width(0.6));
            });
        }
    }

    fn run_full_block(&mut self) {
        use ix_nn::attention::causal_mask;
        use ix_nn::positional::sinusoidal_encoding;
        use ix_nn::transformer::TransformerStack;
        use rand::Rng;
        let mut rng = rand::rng();

        // Ensure d_model divisible by n_heads
        let d = (self.d_model / self.n_heads) * self.n_heads;
        if d == 0 {
            self.status = "d_model must be >= n_heads".into();
            return;
        }

        let stack = TransformerStack::new(self.n_layers, d, self.n_heads, self.d_ff, 42);
        let mask = if self.use_causal {
            Some(causal_mask(self.seq_len))
        } else {
            None
        };

        // Random input + sinusoidal PE
        let pe = sinusoidal_encoding(self.seq_len, d);
        let input = Array3::from_shape_fn((1, self.seq_len, d), |(_, i, j)| {
            rng.random_range(-0.5..0.5) + pe[[i, j]]
        });

        let output = stack.forward(&input, mask.as_ref());

        // Project to 2D for visualization (just take first 2 dims)
        self.input_data = (0..self.seq_len)
            .map(|i| [input[[0, i, 0]], input[[0, i, 1]]])
            .collect();
        self.output_data = (0..self.seq_len)
            .map(|i| [output[[0, i, 0]], output[[0, i, 1]]])
            .collect();

        let in_norm: f64 = input.mapv(|v| v * v).sum().sqrt();
        let out_norm: f64 = output.mapv(|v| v * v).sum().sqrt();
        self.status = format!(
            "{}-layer transformer: seq={}, d_model={}, heads={}, d_ff={} | input L2={:.2}, output L2={:.2}",
            self.n_layers, self.seq_len, d, self.n_heads, self.d_ff, in_norm, out_norm
        );
    }
}
