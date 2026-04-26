mod app;
mod reader;
mod state;
mod views;

use app::App;
use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::prelude::*;
use std::io::stdout;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Determine state directory — default to ../Demerzel/state relative to ix repo
    let state_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            manifest.join("../../governance/demerzel/state")
        });

    if !state_dir.exists() {
        eprintln!("State directory not found: {}", state_dir.display());
        eprintln!("Usage: ix-dashboard [state_dir]");
        eprintln!("Default: governance/demerzel/state (relative to ix repo)");
        std::process::exit(1);
    }

    let mut app = App::new(state_dir);

    // Setup terminal
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout());
    let mut terminal = Terminal::new(backend)?;

    // Main loop
    loop {
        terminal.draw(|frame| ui(frame, &app))?;

        if event::poll(std::time::Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') => app.should_quit = true,
                    KeyCode::Char('r') => app.refresh(),
                    KeyCode::Tab => {
                        if key.modifiers.contains(KeyModifiers::SHIFT) {
                            app.prev_tab();
                        } else {
                            app.next_tab();
                        }
                    }
                    KeyCode::BackTab => app.prev_tab(),
                    KeyCode::Up => app.select_prev(),
                    KeyCode::Down => app.select_next(),
                    _ => {}
                }
            }
        }

        if app.should_quit {
            break;
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;

    Ok(())
}

fn ui(frame: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // tabs
            Constraint::Fill(1),   // main content
            Constraint::Length(1), // status bar
        ])
        .split(frame.area());

    views::render_tabs(app, chunks[0], frame.buffer_mut());

    match app.current_tab {
        app::Tab::Beliefs => views::render_beliefs(app, chunks[1], frame.buffer_mut()),
        app::Tab::Evolution => views::render_evolution(app, chunks[1], frame.buffer_mut()),
        app::Tab::Pdca => views::render_pdca(app, chunks[1], frame.buffer_mut()),
        app::Tab::Conscience => views::render_conscience(app, chunks[1], frame.buffer_mut()),
    }

    views::render_status(app, chunks[2], frame.buffer_mut());
}
