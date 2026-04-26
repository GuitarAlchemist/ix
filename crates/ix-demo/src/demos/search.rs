use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints, Points};

pub struct SearchDemo {
    grid_size: usize,
    obstacle_pct: f64,
    start: [usize; 2],
    goal: [usize; 2],
    path: Vec<[f64; 2]>,
    obstacles: Vec<[f64; 2]>,
    explored: Vec<[f64; 2]>,
    status: String,
}

impl Default for SearchDemo {
    fn default() -> Self {
        Self {
            grid_size: 20,
            obstacle_pct: 0.25,
            start: [0, 0],
            goal: [19, 19],
            path: Vec::new(),
            obstacles: Vec::new(),
            explored: Vec::new(),
            status: "Generate grid and search".into(),
        }
    }
}

impl SearchDemo {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("A* Pathfinding (ix-search)");

        ui.horizontal(|ui| {
            ui.label("Grid:");
            ui.add(egui::Slider::new(&mut self.grid_size, 5..=40));
            ui.label("Obstacles:");
            ui.add(egui::Slider::new(&mut self.obstacle_pct, 0.0..=0.4));
        });

        if ui.button("Generate & Search").clicked() {
            self.goal = [self.grid_size - 1, self.grid_size - 1];
            self.run();
        }

        ui.label(&self.status);

        let gs = self.grid_size as f64;
        Plot::new("search_plot")
            .height(500.0)
            .data_aspect(1.0)
            .include_x(-1.0)
            .include_x(gs)
            .include_y(-1.0)
            .include_y(gs)
            .show(ui, |plot_ui| {
                // Obstacles
                if !self.obstacles.is_empty() {
                    let pts: PlotPoints = self.obstacles.iter().copied().collect();
                    plot_ui.points(
                        Points::new(pts)
                            .radius(5.0)
                            .color(egui::Color32::from_rgb(80, 80, 80))
                            .name("Obstacles")
                            .shape(egui_plot::MarkerShape::Square),
                    );
                }

                // Explored
                if !self.explored.is_empty() {
                    let pts: PlotPoints = self.explored.iter().copied().collect();
                    plot_ui.points(
                        Points::new(pts)
                            .radius(3.0)
                            .color(egui::Color32::from_rgba_premultiplied(100, 150, 255, 80))
                            .name("Explored"),
                    );
                }

                // Path
                if !self.path.is_empty() {
                    let data = self.path.clone();
                    let pts1: PlotPoints = data.iter().copied().collect();
                    let pts2: PlotPoints = data.iter().copied().collect();
                    plot_ui.line(
                        Line::new(pts1)
                            .name("Path")
                            .width(3.0)
                            .color(egui::Color32::YELLOW),
                    );
                    plot_ui.points(Points::new(pts2).radius(5.0).color(egui::Color32::YELLOW));
                }

                // Start/Goal
                plot_ui.points(
                    Points::new(PlotPoints::new(vec![[
                        self.start[0] as f64,
                        self.start[1] as f64,
                    ]]))
                    .radius(10.0)
                    .color(egui::Color32::GREEN)
                    .name("Start"),
                );
                plot_ui.points(
                    Points::new(PlotPoints::new(vec![[
                        self.goal[0] as f64,
                        self.goal[1] as f64,
                    ]]))
                    .radius(10.0)
                    .color(egui::Color32::RED)
                    .name("Goal"),
                );
            });
    }

    fn run(&mut self) {
        use rand::Rng;
        let mut rng = rand::rng();
        let n = self.grid_size;

        // Generate obstacles
        let mut blocked = vec![vec![false; n]; n];
        self.obstacles.clear();
        for (r, row) in blocked.iter_mut().enumerate() {
            for (c, cell) in row.iter_mut().enumerate() {
                if (r == self.start[0] && c == self.start[1])
                    || (r == self.goal[0] && c == self.goal[1])
                {
                    continue;
                }
                if rng.random_range(0.0..1.0) < self.obstacle_pct {
                    *cell = true;
                    self.obstacles.push([r as f64, c as f64]);
                }
            }
        }

        // A* search
        use std::cmp::Reverse;
        use std::collections::{BinaryHeap, HashMap};

        let heuristic = |pos: [usize; 2]| -> f64 {
            let dr = pos[0] as f64 - self.goal[0] as f64;
            let dc = pos[1] as f64 - self.goal[1] as f64;
            (dr * dr + dc * dc).sqrt()
        };

        let mut open = BinaryHeap::new();
        let mut came_from: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
        let mut g_score: HashMap<(usize, usize), f64> = HashMap::new();
        let mut explored_set = Vec::new();

        let start = (self.start[0], self.start[1]);
        let goal = (self.goal[0], self.goal[1]);
        g_score.insert(start, 0.0);
        open.push(Reverse((ordered_float(heuristic(self.start)), start)));

        let mut found = false;
        while let Some(Reverse((_, current))) = open.pop() {
            if current == goal {
                found = true;
                break;
            }
            explored_set.push([current.0 as f64, current.1 as f64]);

            let cur_g = g_score[&current];
            // 4-directional neighbors
            let neighbors: Vec<(usize, usize)> = [
                (current.0.wrapping_sub(1), current.1),
                (current.0 + 1, current.1),
                (current.0, current.1.wrapping_sub(1)),
                (current.0, current.1 + 1),
            ]
            .iter()
            .filter(|&&(r, c)| r < n && c < n && !blocked[r][c])
            .copied()
            .collect();

            for next in neighbors {
                let tentative_g = cur_g + 1.0;
                if tentative_g < *g_score.get(&next).unwrap_or(&f64::INFINITY) {
                    came_from.insert(next, current);
                    g_score.insert(next, tentative_g);
                    let f = tentative_g + heuristic([next.0, next.1]);
                    open.push(Reverse((ordered_float(f), next)));
                }
            }
        }

        self.explored = explored_set;

        if found {
            // Reconstruct path
            let mut path = vec![[goal.0 as f64, goal.1 as f64]];
            let mut cur = goal;
            while let Some(&prev) = came_from.get(&cur) {
                path.push([prev.0 as f64, prev.1 as f64]);
                cur = prev;
            }
            path.reverse();
            self.status = format!(
                "Path found! Length: {}, Explored: {} nodes",
                path.len(),
                self.explored.len()
            );
            self.path = path;
        } else {
            self.path.clear();
            self.status = format!("No path found. Explored {} nodes.", self.explored.len());
        }
    }
}

// Helper for BinaryHeap ordering with f64
fn ordered_float(f: f64) -> u64 {
    let bits = f.to_bits();
    if bits >> 63 == 1 {
        !bits
    } else {
        bits ^ (1 << 63)
    }
}
