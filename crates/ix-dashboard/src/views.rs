use crate::app::{App, Tab};
use crate::state::TetraValue;
use ratatui::prelude::*;
use ratatui::widgets::*;

/// Color for tetravalent values
fn tetra_color(value: &TetraValue) -> Color {
    match value {
        TetraValue::True => Color::Green,
        TetraValue::False => Color::Red,
        TetraValue::Unknown => Color::Yellow,
        TetraValue::Contradictory => Color::Magenta,
    }
}

/// Render the tab bar
pub fn render_tabs(app: &App, area: Rect, buf: &mut Buffer) {
    let titles: Vec<Line> = Tab::ALL
        .iter()
        .map(|t| {
            let style = if *t == app.current_tab {
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::DarkGray)
            };
            Line::from(Span::styled(t.title(), style))
        })
        .collect();

    let tabs = Tabs::new(titles)
        .block(Block::default().borders(Borders::BOTTOM))
        .highlight_style(Style::default().fg(Color::Cyan))
        .select(
            Tab::ALL
                .iter()
                .position(|t| *t == app.current_tab)
                .unwrap_or(0),
        );

    tabs.render(area, buf);
}

/// Render the beliefs panel
pub fn render_beliefs(app: &App, area: Rect, buf: &mut Buffer) {
    let header = Row::new(vec!["Val", "Proposition", "Conf", "Updated"])
        .style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .bottom_margin(1);

    let rows: Vec<Row> = app
        .data
        .beliefs
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let color = tetra_color(&b.value);
            let style = if i == app.selected_row {
                Style::default().bg(Color::DarkGray)
            } else {
                Style::default()
            };

            Row::new(vec![
                Cell::from(b.value.to_string()).style(Style::default().fg(color)),
                Cell::from(b.proposition.clone()),
                Cell::from(format!("{:.2}", b.confidence)),
                Cell::from(b.updated_at.clone().unwrap_or_default()),
            ])
            .style(style)
        })
        .collect();

    let widths = [
        Constraint::Length(3),
        Constraint::Fill(1),
        Constraint::Length(6),
        Constraint::Length(12),
    ];

    let table = Table::new(rows, widths).header(header).block(
        Block::default()
            .title(" Beliefs ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan)),
    );

    Widget::render(table, area, buf);
}

/// Render the evolution panel
pub fn render_evolution(app: &App, area: Rect, buf: &mut Buffer) {
    let header = Row::new(vec!["Artifact", "Type", "Events", "Compliance", "Action"])
        .style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .bottom_margin(1);

    let rows: Vec<Row> = app
        .data
        .evolution
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let compliance_str = e
                .compliance_rate
                .map(|r| format!("{:.0}%", r * 100.0))
                .unwrap_or_else(|| "—".to_string());

            let style = if i == app.selected_row {
                Style::default().bg(Color::DarkGray)
            } else {
                Style::default()
            };

            Row::new(vec![
                Cell::from(e.artifact_name.clone()),
                Cell::from(e.artifact_type.clone()),
                Cell::from(format!("{}", e.events.len())),
                Cell::from(compliance_str),
                Cell::from(e.recommendation.clone().unwrap_or_default()),
            ])
            .style(style)
        })
        .collect();

    let widths = [
        Constraint::Fill(1),
        Constraint::Length(14),
        Constraint::Length(7),
        Constraint::Length(10),
        Constraint::Length(12),
    ];

    let table = Table::new(rows, widths).header(header).block(
        Block::default()
            .title(" Evolution ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Green)),
    );

    Widget::render(table, area, buf);
}

/// Render the PDCA panel
pub fn render_pdca(app: &App, area: Rect, buf: &mut Buffer) {
    let header = Row::new(vec!["Phase", "Name", "Experiment", "Started"])
        .style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .bottom_margin(1);

    let rows: Vec<Row> = app
        .data
        .pdca
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let phase_icon = match p.phase.as_str() {
                "plan" => "P",
                "do" => "D",
                "check" => "C",
                "act" => "A",
                _ => "?",
            };

            let experiment_str = if p.experiment.unwrap_or(false) {
                "yes"
            } else {
                ""
            };

            let style = if i == app.selected_row {
                Style::default().bg(Color::DarkGray)
            } else {
                Style::default()
            };

            Row::new(vec![
                Cell::from(phase_icon),
                Cell::from(p.name.clone()),
                Cell::from(experiment_str),
                Cell::from(p.started_at.clone().unwrap_or_default()),
            ])
            .style(style)
        })
        .collect();

    let widths = [
        Constraint::Length(5),
        Constraint::Fill(1),
        Constraint::Length(10),
        Constraint::Length(12),
    ];

    let table = Table::new(rows, widths).header(header).block(
        Block::default()
            .title(" PDCA Cycles ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Yellow)),
    );

    Widget::render(table, area, buf);
}

/// Render the conscience signals panel
pub fn render_conscience(app: &App, area: Rect, buf: &mut Buffer) {
    let header = Row::new(vec!["ID", "Type", "Weight", "Time"])
        .style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .bottom_margin(1);

    let rows: Vec<Row> = app
        .data
        .signals
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let weight_color = if s.weight >= 0.8 {
                Color::Red
            } else if s.weight >= 0.5 {
                Color::Yellow
            } else {
                Color::Green
            };

            let style = if i == app.selected_row {
                Style::default().bg(Color::DarkGray)
            } else {
                Style::default()
            };

            Row::new(vec![
                Cell::from(s.signal_id.clone()),
                Cell::from(s.signal_type.clone()),
                Cell::from(format!("{:.1}", s.weight)).style(Style::default().fg(weight_color)),
                Cell::from(s.timestamp.clone().unwrap_or_default()),
            ])
            .style(style)
        })
        .collect();

    let widths = [
        Constraint::Length(36),
        Constraint::Fill(1),
        Constraint::Length(6),
        Constraint::Length(22),
    ];

    let table = Table::new(rows, widths).header(header).block(
        Block::default()
            .title(" Conscience Signals ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Magenta)),
    );

    Widget::render(table, area, buf);
}

/// Render the status bar
pub fn render_status(app: &App, area: Rect, buf: &mut Buffer) {
    let beliefs_count = app.data.beliefs.len();
    let evolution_count = app.data.evolution.len();
    let pdca_count = app.data.pdca.len();
    let signal_count = app.data.signals.len();

    let status = format!(
        " Beliefs: {}  Evolution: {}  PDCA: {}  Signals: {}  |  Tab/Shift+Tab: switch  Up/Down: select  r: refresh  q: quit",
        beliefs_count, evolution_count, pdca_count, signal_count
    );

    let paragraph = Paragraph::new(status).style(Style::default().fg(Color::DarkGray));

    paragraph.render(area, buf);
}
