use crate::reader;
use crate::state::DashboardData;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Tab {
    Beliefs,
    Evolution,
    Pdca,
    Conscience,
}

impl Tab {
    pub const ALL: [Tab; 4] = [Tab::Beliefs, Tab::Evolution, Tab::Pdca, Tab::Conscience];

    pub fn title(&self) -> &'static str {
        match self {
            Tab::Beliefs => "Beliefs",
            Tab::Evolution => "Evolution",
            Tab::Pdca => "PDCA",
            Tab::Conscience => "Conscience",
        }
    }

    pub fn next(&self) -> Tab {
        match self {
            Tab::Beliefs => Tab::Evolution,
            Tab::Evolution => Tab::Pdca,
            Tab::Pdca => Tab::Conscience,
            Tab::Conscience => Tab::Beliefs,
        }
    }

    pub fn prev(&self) -> Tab {
        match self {
            Tab::Beliefs => Tab::Conscience,
            Tab::Evolution => Tab::Beliefs,
            Tab::Pdca => Tab::Evolution,
            Tab::Conscience => Tab::Pdca,
        }
    }
}

pub struct App {
    pub current_tab: Tab,
    pub data: DashboardData,
    pub state_dir: PathBuf,
    pub should_quit: bool,
    pub selected_row: usize,
}

impl App {
    pub fn new(state_dir: PathBuf) -> Self {
        let data = reader::load_all(&state_dir);
        Self {
            current_tab: Tab::Beliefs,
            data,
            state_dir,
            should_quit: false,
            selected_row: 0,
        }
    }

    pub fn refresh(&mut self) {
        self.data = reader::load_all(&self.state_dir);
        self.selected_row = 0;
    }

    pub fn next_tab(&mut self) {
        self.current_tab = self.current_tab.next();
        self.selected_row = 0;
    }

    pub fn prev_tab(&mut self) {
        self.current_tab = self.current_tab.prev();
        self.selected_row = 0;
    }

    pub fn row_count(&self) -> usize {
        match self.current_tab {
            Tab::Beliefs => self.data.beliefs.len(),
            Tab::Evolution => self.data.evolution.len(),
            Tab::Pdca => self.data.pdca.len(),
            Tab::Conscience => self.data.signals.len(),
        }
    }

    pub fn select_next(&mut self) {
        let count = self.row_count();
        if count > 0 {
            self.selected_row = (self.selected_row + 1).min(count - 1);
        }
    }

    pub fn select_prev(&mut self) {
        self.selected_row = self.selected_row.saturating_sub(1);
    }
}
