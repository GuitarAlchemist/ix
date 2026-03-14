use eframe::egui;
use egui_plot::{Plot, Line, PlotPoints};

pub struct SignalDemo {
    freq1: f64,
    freq2: f64,
    noise_level: f64,
    n_samples: usize,
    time_data: Vec<[f64; 2]>,
    freq_data: Vec<[f64; 2]>,
    filtered_data: Vec<[f64; 2]>,
}

impl Default for SignalDemo {
    fn default() -> Self {
        Self {
            freq1: 5.0,
            freq2: 20.0,
            noise_level: 0.3,
            n_samples: 256,
            time_data: Vec::new(),
            freq_data: Vec::new(),
            filtered_data: Vec::new(),
        }
    }
}

impl SignalDemo {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Signal Processing (machin-signal)");
        ui.label("Generate a signal with two frequencies, compute FFT, and low-pass filter.");

        ui.horizontal(|ui| {
            ui.label("Freq 1 (Hz):"); ui.add(egui::Slider::new(&mut self.freq1, 1.0..=50.0));
            ui.label("Freq 2 (Hz):"); ui.add(egui::Slider::new(&mut self.freq2, 1.0..=50.0));
        });
        ui.horizontal(|ui| {
            ui.label("Noise:"); ui.add(egui::Slider::new(&mut self.noise_level, 0.0..=2.0));
            ui.label("Samples:"); ui.add(egui::Slider::new(&mut self.n_samples, 64..=1024));
        });

        if ui.button("Analyze").clicked() {
            self.analyze();
        }

        if !self.time_data.is_empty() {
            Plot::new("signal_time").height(200.0).show(ui, |plot_ui| {
                let pts: PlotPoints = self.time_data.iter().copied().collect();
                plot_ui.line(Line::new(pts).name("Signal").width(1.0));
                if !self.filtered_data.is_empty() {
                    let fpts: PlotPoints = self.filtered_data.iter().copied().collect();
                    plot_ui.line(Line::new(fpts).name("Filtered").width(2.0).color(egui::Color32::YELLOW));
                }
            });

            ui.label("FFT Magnitude Spectrum:");
            Plot::new("signal_freq").height(200.0).show(ui, |plot_ui| {
                let pts: PlotPoints = self.freq_data.iter().copied().collect();
                plot_ui.line(Line::new(pts).name("|FFT|").width(1.5).color(egui::Color32::from_rgb(100, 200, 255)));
            });
        }
    }

    fn analyze(&mut self) {
        use rand::Rng;
        let mut rng = rand::rng();
        let n = self.n_samples;
        let fs = 100.0; // sample rate

        // Generate signal
        let signal: Vec<f64> = (0..n).map(|i| {
            let t = i as f64 / fs;
            
            (2.0 * std::f64::consts::PI * self.freq1 * t).sin()
                + 0.5 * (2.0 * std::f64::consts::PI * self.freq2 * t).sin()
                + rng.random_range(-self.noise_level..self.noise_level)
        }).collect();

        self.time_data = signal.iter().enumerate()
            .map(|(i, &v)| [i as f64 / fs, v])
            .collect();

        // FFT using machin-signal
        use machin_signal::fft::{Complex, fft};
        let complex_input: Vec<Complex> = signal.iter()
            .map(|&v| Complex::from_real(v))
            .collect();
        let spectrum = fft(&complex_input);
        let half = n / 2;
        self.freq_data = (0..half).map(|i| {
            let freq = i as f64 * fs / n as f64;
            let mag = spectrum[i].magnitude() / n as f64 * 2.0;
            [freq, mag]
        }).collect();

        // Simple moving average as low-pass filter
        let window = 5;
        let mut filtered = vec![0.0; signal.len()];
        for (i, slot) in filtered.iter_mut().enumerate() {
            let start = i.saturating_sub(window / 2);
            let end = (i + window / 2 + 1).min(signal.len());
            let sum: f64 = signal[start..end].iter().sum();
            *slot = sum / (end - start) as f64;
        }

        self.filtered_data = filtered.iter().enumerate()
            .map(|(i, &v)| [i as f64 / fs, v])
            .collect();
    }
}
