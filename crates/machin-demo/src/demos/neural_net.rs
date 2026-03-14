use eframe::egui;
use egui_plot::{Plot, Line, PlotPoints};

pub struct NeuralNetDemo {
    layers: String,
    learning_rate: f64,
    epochs: usize,
    loss_history: Vec<f64>,
    status: String,
}

impl Default for NeuralNetDemo {
    fn default() -> Self {
        Self {
            layers: "2, 8, 4, 1".into(),
            learning_rate: 0.1,
            epochs: 200,
            loss_history: Vec::new(),
            status: "Ready".into(),
        }
    }
}

impl NeuralNetDemo {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Neural Network (machin-nn)");
        ui.label("XOR problem — trains a small MLP on the XOR dataset.");

        ui.horizontal(|ui| {
            ui.label("Layers:");
            ui.text_edit_singleline(&mut self.layers);
        });
        ui.horizontal(|ui| {
            ui.label("Learning rate:");
            ui.add(egui::Slider::new(&mut self.learning_rate, 0.01..=1.0).logarithmic(true));
            ui.label("Epochs:");
            ui.add(egui::Slider::new(&mut self.epochs, 10..=2000));
        });

        if ui.button("Train").clicked() {
            self.train();
        }

        ui.label(&self.status);

        if !self.loss_history.is_empty() {
            Plot::new("nn_loss").height(300.0).show(ui, |plot_ui| {
                let pts: PlotPoints = self.loss_history.iter().enumerate()
                    .map(|(i, &l)| [i as f64, l])
                    .collect();
                plot_ui.line(Line::new(pts).name("Loss").width(2.0));
            });
        }
    }

    fn train(&mut self) {
        use ndarray::{array, Array2};

        // XOR dataset
        let x = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let y = array![[0.0], [1.0], [1.0], [0.0]];

        let layer_sizes: Vec<usize> = self.layers.split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();

        if layer_sizes.len() < 2 {
            self.status = "Need at least 2 layer sizes".into();
            return;
        }

        // Manual MLP training with backprop (demonstrates the concepts in machin-nn)
        let mut weights: Vec<Array2<f64>> = Vec::new();
        let mut biases: Vec<ndarray::Array1<f64>> = Vec::new();
        use rand::Rng;
        let mut rng = rand::rng();

        for i in 0..layer_sizes.len()-1 {
            let w = Array2::from_shape_fn(
                (layer_sizes[i], layer_sizes[i+1]),
                |_| rng.random_range(-1.0..1.0)
            );
            let b = ndarray::Array1::zeros(layer_sizes[i+1]);
            weights.push(w);
            biases.push(b);
        }

        self.loss_history.clear();
        let lr = self.learning_rate;

        for _epoch in 0..self.epochs {
            // Forward pass
            let mut activations_list: Vec<Array2<f64>> = vec![x.clone()];
            let mut current = x.clone();
            for i in 0..weights.len() {
                let z = current.dot(&weights[i]) + &biases[i];
                current = z.mapv(|v| 1.0 / (1.0 + (-v).exp())); // sigmoid
                activations_list.push(current.clone());
            }

            // Loss (MSE)
            let output = activations_list.last().unwrap();
            let error = output - &y;
            let loss = error.mapv(|v| v * v).mean().unwrap_or(0.0);
            self.loss_history.push(loss);

            // Backprop
            let mut delta = &error * &output.mapv(|v| v * (1.0 - v)) * 2.0 / 4.0;
            for i in (0..weights.len()).rev() {
                let grad_w = activations_list[i].t().dot(&delta);
                let grad_b = delta.sum_axis(ndarray::Axis(0));
                weights[i] = &weights[i] - &(grad_w * lr);
                biases[i] = &biases[i] - &(grad_b * lr);
                if i > 0 {
                    let act = &activations_list[i];
                    delta = delta.dot(&weights[i].t()) * &act.mapv(|v| v * (1.0 - v));
                }
            }
        }

        let final_loss = self.loss_history.last().copied().unwrap_or(0.0);
        self.status = format!("Trained {} epochs. Final loss: {:.6}", self.epochs, final_loss);
    }
}
