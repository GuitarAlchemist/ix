use crate::state::{ConsoleState, NodeHealth, NodeId};
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Widget};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IcicleCell {
    pub node_id: NodeId,
    pub rel_depth: usize,
    pub abs_depth: usize,
    pub row_start: u16,
    pub row_span: u16,
    pub col_start: u16,
    pub col_width: u16,
    pub is_selected: bool,
    pub is_zoom_root: bool,
    pub hidden_siblings_after: usize,
}

#[derive(Debug, Clone, Default)]
pub struct ConsoleLayoutPlan {
    pub area: Rect,
    pub banner_area: Rect,
    pub breadcrumb_area: Rect,
    pub left_area: Rect,
    pub right_area: Rect,
    pub footer_area: Rect,
    pub degraded_narrow: bool,
    pub icicle_cells: Vec<IcicleCell>,
}

/// Compute the deterministic Zoomable Icicle layout plan for the given terminal `area`.
/// Uses the exact same recursive partition rule regardless of the current zoom depth.
pub fn compute_layout_plan(state: &ConsoleState, area: Rect) -> ConsoleLayoutPlan {
    if area.width < 36 || area.height < 8 {
        return ConsoleLayoutPlan {
            area,
            banner_area: area,
            breadcrumb_area: Rect::default(),
            left_area: area,
            right_area: Rect::default(),
            footer_area: Rect::default(),
            degraded_narrow: true,
            icicle_cells: Vec::new(),
        };
    }

    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Fill(1),
            Constraint::Length(1),
        ])
        .split(area);

    let panes = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(52), Constraint::Percentage(48)])
        .split(vertical[2]);

    let left_area = panes[0];
    let right_area = panes[1];

    let inner_left = inner_rect(left_area);
    let mut icicle_cells = Vec::new();

    if inner_left.width >= 8 && inner_left.height >= 1 {
        if let Some(root_id) = state
            .zoom_root_id
            .as_ref()
            .or_else(|| state.roots.first())
            .filter(|id| state.nodes.contains_key(*id))
        {
            let max_cols = ((inner_left.width / 12) as usize).clamp(1, 4);
            let col_w = inner_left.width / (max_cols as u16);
            partition_icicle_recursive(
                state,
                root_id,
                0,
                max_cols,
                inner_left.y,
                inner_left.height,
                inner_left.x,
                col_w,
                inner_left.x + inner_left.width,
                &mut icicle_cells,
            );
        }
    }

    ConsoleLayoutPlan {
        area,
        banner_area: vertical[0],
        breadcrumb_area: vertical[1],
        left_area,
        right_area,
        footer_area: vertical[3],
        degraded_narrow: false,
        icicle_cells,
    }
}

#[allow(clippy::too_many_arguments)]
fn partition_icicle_recursive(
    state: &ConsoleState,
    node_id: &str,
    rel_depth: usize,
    max_cols: usize,
    row_start: u16,
    row_span: u16,
    col_base_x: u16,
    col_w: u16,
    right_edge: u16,
    out: &mut Vec<IcicleCell>,
) {
    if row_span == 0 || rel_depth >= max_cols {
        return;
    }
    let Some(entry) = state.nodes.get(node_id) else {
        return;
    };

    let col_start = col_base_x + (rel_depth as u16) * col_w;
    let col_width = if rel_depth + 1 == max_cols {
        right_edge.saturating_sub(col_start)
    } else {
        col_w
    };
    if col_width == 0 {
        return;
    }

    let is_selected = state.selected_id.as_deref() == Some(node_id);
    let is_zoom_root = state.zoom_root_id.as_deref() == Some(node_id);

    out.push(IcicleCell {
        node_id: node_id.to_string(),
        rel_depth,
        abs_depth: entry.depth,
        row_start,
        row_span,
        col_start,
        col_width,
        is_selected,
        is_zoom_root,
        hidden_siblings_after: 0,
    });

    if rel_depth + 1 >= max_cols || entry.children.is_empty() {
        return;
    }

    // Filter active children (keep selected child even if tombstoned)
    let active_children: Vec<&NodeId> = entry
        .children
        .iter()
        .filter(|cid| {
            state
                .nodes
                .get(*cid)
                .map(|c| c.removed_at_ms.is_none() || state.selected_id.as_ref() == Some(*cid))
                .unwrap_or(false)
        })
        .collect();

    if active_children.is_empty() {
        return;
    }

    let max_visible_children = (row_span as usize).min(active_children.len());
    if max_visible_children == 0 {
        return;
    }

    // Ensure the selected node (or ancestor of the selected node) is included in visible slice
    let visible_slice: Vec<&NodeId> = active_children
        .iter()
        .take(max_visible_children)
        .copied()
        .collect();
    let hidden_count = active_children.len().saturating_sub(max_visible_children);

    let weights: Vec<usize> = visible_slice
        .iter()
        .map(|cid| {
            state
                .nodes
                .get(*cid)
                .map(|n| n.subtree_weight.max(1))
                .unwrap_or(1)
        })
        .collect();
    let total_weight: usize = weights.iter().sum::<usize>().max(1);

    // Every visible child receives at least 1 row; remaining rows are distributed proportionally
    let extra_rows = (row_span as usize).saturating_sub(max_visible_children);
    let mut spans: Vec<u16> = vec![1; max_visible_children];
    let mut allocated_extra = 0usize;

    if extra_rows > 0 {
        for (i, w) in weights.iter().enumerate() {
            let share = (extra_rows * (*w)) / total_weight;
            spans[i] += share as u16;
            allocated_extra += share;
        }
        let mut remainder = extra_rows.saturating_sub(allocated_extra);
        let span_len = spans.len();
        let mut idx = 0usize;
        while remainder > 0 && span_len > 0 {
            spans[idx % span_len] += 1;
            remainder -= 1;
            idx += 1;
        }
    }

    let mut curr_y = row_start;
    for (i, child_id) in visible_slice.iter().enumerate() {
        let span = spans[i];
        let prev_len = out.len();
        partition_icicle_recursive(
            state,
            child_id,
            rel_depth + 1,
            max_cols,
            curr_y,
            span,
            col_base_x,
            col_w,
            right_edge,
            out,
        );
        if i + 1 == visible_slice.len() && hidden_count > 0 {
            if let Some(cell) = out.get_mut(prev_len) {
                cell.hidden_siblings_after = hidden_count;
            }
        }
        curr_y = curr_y.saturating_add(span);
    }
}

pub fn render_console(state: &ConsoleState, area: Rect, buf: &mut Buffer) -> ConsoleLayoutPlan {
    let plan = compute_layout_plan(state, area);
    if plan.area.width == 0 || plan.area.height == 0 {
        return plan;
    }

    if plan.degraded_narrow {
        let sel = state.selected_id.as_deref().unwrap_or("-");
        let msg = format!("SYNTHETIC IX-CONSOLE | sel={sel}");
        write_ascii_clipped(
            buf,
            area.x,
            area.y,
            area.width,
            &msg,
            Style::default().fg(Color::Yellow),
        );
        if area.height > 1 {
            let zoom = state.zoom_root_id.as_deref().unwrap_or("-");
            let line2 = format!("zoom={zoom} ({}x{} narrow)", area.width, area.height);
            write_ascii_clipped(
                buf,
                area.x,
                area.y + 1,
                area.width,
                &line2,
                Style::default().fg(Color::Gray),
            );
        }
        return plan;
    }

    let cov = state.coverage_summary();

    // 1. Top Synthetic Banner + Coverage bar
    let banner_text = format!(
        " {} | COV total={} fresh={} stale={} unkn={} gaps={}",
        state.fixture_banner,
        cov.total_nodes,
        cov.fresh_nodes,
        cov.stale_nodes,
        cov.unknown_nodes,
        cov.gaps_detected
    );
    write_ascii_clipped(
        buf,
        plan.banner_area.x,
        plan.banner_area.y,
        plan.banner_area.width,
        &banner_text,
        Style::default()
            .fg(Color::Black)
            .bg(Color::Yellow)
            .add_modifier(Modifier::BOLD),
    );

    // 2. Breadcrumb bar (scale-invariant path to current zoom root)
    let crumbs = state.breadcrumbs();
    let crumb_str = if crumbs.is_empty() {
        " <empty-tree>".to_string()
    } else {
        format!(" BREADCRUMB (Zoom Path): {}", crumbs.join(" > "))
    };
    write_ascii_clipped(
        buf,
        plan.breadcrumb_area.x,
        plan.breadcrumb_area.y,
        plan.breadcrumb_area.width,
        &crumb_str,
        Style::default().fg(Color::Cyan),
    );

    // 3. Left Pane: Zoomable Icicle ASCII
    render_left_icicle(state, &plan, buf);

    // 4. Right Pane: Single Selected Node Detail & Bounded Escaped Logs
    render_right_detail(state, &plan, &cov, buf);

    // 5. Footer help bar
    let footer_text = " KEYS: j/k/Up/Down=Select | Enter/l=ZoomIn | Backspace/h=Back | 0=Root | f=FollowLogs | PgUp/PgDn=Scroll | q=Quit";
    write_ascii_clipped(
        buf,
        plan.footer_area.x,
        plan.footer_area.y,
        plan.footer_area.width,
        footer_text,
        Style::default().fg(Color::DarkGray),
    );

    plan
}

fn render_left_icicle(state: &ConsoleState, plan: &ConsoleLayoutPlan, buf: &mut Buffer) {
    let zoom_label = state.zoom_root_id.as_deref().unwrap_or("<none>");
    let block = Block::default()
        .title(format!(" Zoomable Icicle [root: {zoom_label}] "))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));
    block.render(plan.left_area, buf);

    let inner = inner_rect(plan.left_area);
    if inner.width == 0 || inner.height == 0 {
        return;
    }

    if plan.icicle_cells.is_empty() {
        write_ascii_clipped(
            buf,
            inner.x,
            inner.y,
            inner.width,
            " (empty tree - no synthetic nodes loaded)",
            Style::default().fg(Color::DarkGray),
        );
        return;
    }

    for cell in &plan.icicle_cells {
        let Some(entry) = state.nodes.get(&cell.node_id) else {
            continue;
        };
        let eff_health = state.effective_health(&cell.node_id);
        let style = cell_style(&eff_health, cell.is_selected, entry.removed_at_ms.is_some());

        for dy in 0..cell.row_span {
            let y = cell.row_start + dy;
            if y >= inner.y + inner.height {
                break;
            }
            let text = if dy == 0 {
                let marker = if cell.is_selected {
                    ">>"
                } else if cell.is_zoom_root {
                    "@ "
                } else {
                    "  "
                };
                let badge = if entry.removed_at_ms.is_some() {
                    "TOMB"
                } else {
                    eff_health.badge()
                };
                format!(
                    "|{marker}[{badge}] {} (d{})",
                    entry.spec.label, cell.abs_depth
                )
            } else if dy + 1 == cell.row_span && cell.hidden_siblings_after > 0 {
                format!("|  ... +{} sibling(s)", cell.hidden_siblings_after)
            } else if dy == 1 && cell.row_span >= 2 {
                format!("|   id:{}", entry.spec.id)
            } else {
                "|".to_string()
            };
            write_ascii_clipped(buf, cell.col_start, y, cell.col_width, &text, style);
        }
    }
}

fn render_right_detail(
    state: &ConsoleState,
    plan: &ConsoleLayoutPlan,
    cov: &crate::state::CoverageSummary,
    buf: &mut Buffer,
) {
    let block = Block::default()
        .title(" Selected Node Context (1 Node Focus) ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Green));

    let inner = inner_rect(plan.right_area);
    block.render(plan.right_area, buf);

    if inner.width == 0 || inner.height == 0 {
        return;
    }

    let Some(sel_id) = state.selected_id.as_ref() else {
        let p = Paragraph::new("No node selected. Tree is empty.");
        p.render(inner, buf);
        return;
    };

    let Some(entry) = state.nodes.get(sel_id) else {
        let p = Paragraph::new(format!("Selected node '{sel_id}' not found in index."));
        p.render(inner, buf);
        return;
    };

    let eff_health = state.effective_health(sel_id);
    let visual_span_str = plan
        .icicle_cells
        .iter()
        .find(|c| &c.node_id == sel_id)
        .map(|c| format!("{}r x {}c (rel_d={})", c.row_span, c.col_width, c.rel_depth))
        .unwrap_or_else(|| "off-viewport (anchored)".to_string());

    let progress_str = entry
        .spec
        .progress_bp
        .map(|bp| format!("{} {:5.1}%", ascii_bar(bp, 10), (bp as f64) / 100.0))
        .unwrap_or_else(|| "n/a (unreported)".to_string());

    let quality_str = entry
        .spec
        .quality_bp
        .map(|bp| format!("{:5.1}% ({} bp)", (bp as f64) / 100.0, bp))
        .unwrap_or_else(|| "n/a (unverified)".to_string());

    let health_detail = match &eff_health {
        NodeHealth::Stale { silent_ms } => format!(
            "STALE (silent {}ms > budget {}ms; Silence != Idle)",
            silent_ms, entry.spec.stale_after_ms
        ),
        NodeHealth::Unknown(reason) => {
            format!("UNKNOWN ({}; Silence != Idle)", reason.short_code())
        }
        other => format!(
            "{:?} (hb={:?}ms, budget={}ms)",
            other, entry.spec.last_heartbeat_ms, entry.spec.stale_after_ms
        ),
    };

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(vec![
        Span::styled("TAG: ", Style::default().fg(Color::Yellow)),
        Span::styled(
            "[SYNTHETIC FIXTURE DATA - READ ONLY]",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
    ]));

    lines.push(Line::from(format!(
        "Node ID   : {} ({})",
        entry.spec.id, entry.spec.label
    )));
    lines.push(Line::from(format!(
        "Parent ID : {} | Children: {} | SubtreeW: {}",
        entry.spec.parent_id.as_deref().unwrap_or("<root>"),
        entry.children.len(),
        entry.subtree_weight
    )));

    let inc_line = match &entry.previous_incarnation {
        Some(prev) => format!(
            "Incarnat. : {} (REPLACED prev={} transitions={})",
            entry.spec.incarnation, prev, entry.incarnation_transitions
        ),
        None => format!("Incarnat. : {}", entry.spec.incarnation),
    };
    lines.push(Line::from(inc_line));

    if let Some(removed_ms) = entry.removed_at_ms {
        lines.push(Line::from(Span::styled(
            format!("LIFECYCLE : TOMBSTONED / REMOVED at t={removed_ms}ms (focus anchored)"),
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        )));
    }

    lines.push(Line::from(format!(
        "Task      : {}",
        entry.spec.task_summary
    )));
    lines.push(Line::from(Span::styled(
        format!("State     : {health_detail}"),
        Style::default().fg(health_fg(&eff_health)),
    )));

    // 4 strictly orthogonal dimensions
    lines.push(Line::from(
        "--- Orthogonal Dimensions (Depth != Progress != Quality != Size) ---",
    ));
    lines.push(Line::from(format!(
        "1. Depth    : abs_depth={} | 2. VisualSpan: {}",
        entry.depth, visual_span_str
    )));
    lines.push(Line::from(format!(
        "3. Progress : {} | 4. Quality   : {}",
        progress_str, quality_str
    )));
    lines.push(Line::from(format!(
        "Coverage  : observed={}/{} fresh={} stale={} unknown={} gaps={}",
        cov.observed_nodes,
        cov.total_nodes,
        cov.fresh_nodes,
        cov.stale_nodes,
        cov.unknown_nodes,
        cov.gaps_detected
    )));

    // Evidence section
    lines.push(Line::from(format!(
        "--- Synthetic Evidence ({}) ---",
        entry.spec.evidence.len()
    )));
    if entry.spec.evidence.is_empty() {
        lines.push(Line::from("  (no evidence attached)"));
    } else {
        for ev in entry.spec.evidence.iter().take(2) {
            let tag = if ev.synthetic { "[SYNTH]" } else { "[LIVE]" };
            lines.push(Line::from(format!(
                "  {tag} {}: {} -> {}",
                ev.kind, ev.label, ev.uri
            )));
        }
    }

    // Bounded & escaped logs section
    lines.push(Line::from(format!(
        "--- Bounded Logs (kept={}/{} evicted={} follow={}) ---",
        entry.logs.len(),
        state.max_logs_per_node,
        entry.evicted_log_count,
        if state.follow_logs { "ON" } else { "OFF" }
    )));

    let remaining_rows = (inner.height as usize).saturating_sub(lines.len());
    if remaining_rows > 0 {
        let total_logs = entry.logs.len();
        let start_idx = if state.follow_logs {
            total_logs.saturating_sub(remaining_rows)
        } else {
            entry
                .log_scroll_offset
                .min(total_logs.saturating_sub(remaining_rows))
        };

        for log in entry.logs.iter().skip(start_idx).take(remaining_rows) {
            let esc_flag = if log.was_escaped { "[ESC]" } else { "     " };
            lines.push(Line::from(format!(
                "#{:04} @{:06}ms {esc_flag} {}",
                log.seq, log.timestamp_ms, log.text
            )));
        }
    }

    let paragraph = Paragraph::new(lines);
    paragraph.render(inner, buf);
}

fn inner_rect(r: Rect) -> Rect {
    if r.width <= 2 || r.height <= 2 {
        Rect::default()
    } else {
        Rect::new(r.x + 1, r.y + 1, r.width - 2, r.height - 2)
    }
}

fn ascii_bar(bp: u16, width: usize) -> String {
    let clamped = bp.min(10_000) as usize;
    let filled = (clamped * width) / 10_000;
    let mut s = String::with_capacity(width + 2);
    s.push('[');
    for i in 0..width {
        s.push(if i < filled { '#' } else { '-' });
    }
    s.push(']');
    s
}

fn health_fg(health: &NodeHealth) -> Color {
    match health {
        NodeHealth::Working => Color::Cyan,
        NodeHealth::Idle => Color::Green,
        NodeHealth::Blocked => Color::Yellow,
        NodeHealth::Done => Color::Blue,
        NodeHealth::Failed => Color::Red,
        NodeHealth::Stale { .. } => Color::LightRed,
        NodeHealth::Unknown(_) => Color::Magenta,
    }
}

fn cell_style(health: &NodeHealth, is_selected: bool, is_removed: bool) -> Style {
    let fg = if is_removed {
        Color::Red
    } else {
        health_fg(health)
    };
    if is_selected {
        Style::default()
            .fg(Color::Black)
            .bg(Color::Cyan)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(fg)
    }
}

fn write_ascii_clipped(buf: &mut Buffer, x: u16, y: u16, max_width: u16, text: &str, style: Style) {
    if max_width == 0 || y >= buf.area.y + buf.area.height {
        return;
    }
    let right_limit = (x + max_width).min(buf.area.x + buf.area.width);
    let mut cx = x;
    for ch in text.chars() {
        if cx >= right_limit {
            break;
        }
        let safe_ch = if ch.is_ascii() && !ch.is_ascii_control() {
            ch
        } else {
            '?'
        };
        buf[(cx, y)].set_char(safe_ch).set_style(style);
        cx += 1;
    }
    while cx < right_limit {
        buf[(cx, y)].set_char(' ').set_style(style);
        cx += 1;
    }
}

pub fn buffer_to_ascii_string(buf: &Buffer) -> String {
    let area = buf.area;
    let mut out = String::with_capacity((area.width as usize + 1) * (area.height as usize));
    for y in 0..area.height {
        for x in 0..area.width {
            let symbol = buf[(x, y)].symbol();
            let ch = symbol.chars().next().unwrap_or(' ');
            out.push(ch);
        }
        if y + 1 < area.height {
            out.push('\n');
        }
    }
    out
}
