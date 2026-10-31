#[path = "ix_console/feed.rs"]
mod feed;
#[path = "ix_console/layout.rs"]
mod layout;
#[path = "ix_console/state.rs"]
mod state;

use crossterm::{
    event::{self, Event},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use feed::{build_synthetic_10k_state, load_synthetic_demo_state};
use layout::{buffer_to_ascii_string, compute_layout_plan, render_console};
use ratatui::prelude::*;
use state::{
    action_from_json, action_from_key, action_from_voice, ConsoleAction, ConsoleState, FeedEvent,
    VoiceIntent,
};
use std::io::stdout;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return Ok(());
    }

    if args.iter().any(|a| a == "--bench-10k") {
        let iters = parse_arg_usize(&args, "--iters").unwrap_or(200);
        run_benchmark_10k(iters);
        return Ok(());
    }

    if args.iter().any(|a| a == "--snapshot") {
        let width = parse_arg_u16(&args, "--width").unwrap_or(108);
        let height = parse_arg_u16(&args, "--height").unwrap_or(26);
        let zoom_opt = parse_arg_str(&args, "--zoom");
        let select_opt = parse_arg_str(&args, "--select");
        run_snapshot(width, height, zoom_opt.as_deref(), select_opt.as_deref());
        return Ok(());
    }

    run_interactive_tui()
}

fn print_usage() {
    println!("ix_console — IX Two-Pane Zoomable Icicle ASCII Console (Synthetic Demo v1)");
    println!();
    println!("USAGE:");
    println!("  cargo run -p ix-dashboard --example ix_console");
    println!("  cargo run -p ix-dashboard --example ix_console -- --snapshot [--width 108] [--height 26] [--zoom <id>] [--select <id>]");
    println!(
        "  cargo run --release -p ix-dashboard --example ix_console -- --bench-10k [--iters 200]"
    );
}

fn parse_arg_str(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn parse_arg_u16(args: &[String], flag: &str) -> Option<u16> {
    parse_arg_str(args, flag).and_then(|v| v.parse::<u16>().ok())
}

fn parse_arg_usize(args: &[String], flag: &str) -> Option<usize> {
    parse_arg_str(args, flag).and_then(|v| v.parse::<usize>().ok())
}

fn run_snapshot(width: u16, height: u16, zoom_to: Option<&str>, select: Option<&str>) {
    let mut state = load_synthetic_demo_state();
    if let Some(z) = zoom_to {
        if let Ok(act) = action_from_voice(&VoiceIntent::ZoomToNode {
            target_id: z.to_string(),
        }) {
            state.apply_action(act);
        }
    }
    if let Some(s) = select {
        if let Ok(act) = action_from_voice(&VoiceIntent::SelectNode {
            target_id: s.to_string(),
        }) {
            state.apply_action(act);
        }
    }
    let area = Rect::new(0, 0, width, height);
    let mut buf = Buffer::empty(area);
    let _ = render_console(&state, area, &mut buf);
    println!("{}", buffer_to_ascii_string(&buf));
}

fn run_benchmark_10k(iters: usize) {
    let build_start = Instant::now();
    let mut state = build_synthetic_10k_state(10_000);
    let build_elapsed = build_start.elapsed();

    let area = Rect::new(0, 0, 110, 30);
    let mut reduce_layout_us = Vec::with_capacity(iters);
    let mut input_render_us = Vec::with_capacity(iters);

    for i in 0..iters {
        let t0 = Instant::now();
        state.apply_feed(FeedEvent::AppendLog {
            node_id: "dom.00".to_string(),
            seq: (i + 1) as u64,
            timestamp_ms: 100_000 + (i as u64),
            raw_text: format!("[SYNTHETIC-BENCH] tick={i} ansi=\x1b[32mOK\x1b[0m"),
        });
        state.apply_action(if i % 2 == 0 {
            ConsoleAction::SelectNextVisible
        } else {
            ConsoleAction::SelectPrevVisible
        });
        let plan = compute_layout_plan(&state, area);
        std::hint::black_box(&plan);
        reduce_layout_us.push(t0.elapsed().as_micros() as u64);

        let t1 = Instant::now();
        let cmd = if i % 4 == 0 {
            r#"{"action":"zoom_in"}"#
        } else if i % 4 == 2 {
            r#"{"action":"back"}"#
        } else {
            r#"{"action":"select_next"}"#
        };
        if let Ok(act) = action_from_json(cmd) {
            state.apply_action(act);
        }
        let mut buf = Buffer::empty(area);
        let plan_render = render_console(&state, area, &mut buf);
        std::hint::black_box(&plan_render);
        std::hint::black_box(&buf);
        input_render_us.push(t1.elapsed().as_micros() as u64);
    }

    reduce_layout_us.sort_unstable();
    input_render_us.sort_unstable();

    let rl_p50 = percentile(&reduce_layout_us, 50);
    let rl_p95 = percentile(&reduce_layout_us, 95);
    let rl_max = *reduce_layout_us.last().unwrap_or(&0);

    let ir_p50 = percentile(&input_render_us, 50);
    let ir_p95 = percentile(&input_render_us, 95);
    let ir_max = *input_render_us.last().unwrap_or(&0);

    let approx_bytes = estimate_state_bytes(&state);

    println!("=== IX-CONSOLE-ASCII 10K BENCHMARK (SYNTHETIC FIXTURE) ===");
    println!("nodes_total          : {}", state.nodes.len());
    println!("iterations           : {}", iters);
    println!(
        "build_10k_ms         : {:.3}",
        build_elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "estimated_state_kib  : {:.1}",
        (approx_bytes as f64) / 1024.0
    );
    println!(
        "reduce_layout_ms     : p50={:.3} ms | p95={:.3} ms | max={:.3} ms (target p95 < 5.000 ms)",
        (rl_p50 as f64) / 1000.0,
        (rl_p95 as f64) / 1000.0,
        (rl_max as f64) / 1000.0
    );
    println!(
        "input_to_render_ms   : p50={:.3} ms | p95={:.3} ms | max={:.3} ms (target p95 < 50.000 ms)",
        (ir_p50 as f64) / 1000.0,
        (ir_p95 as f64) / 1000.0,
        (ir_max as f64) / 1000.0
    );
}

fn percentile(sorted: &[u64], pct: usize) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) * pct) / 100;
    sorted[idx]
}

fn estimate_state_bytes(state: &ConsoleState) -> usize {
    let mut bytes = std::mem::size_of::<ConsoleState>();
    for (k, v) in &state.nodes {
        bytes += k.len() + std::mem::size_of::<state::NodeEntry>();
        bytes += v.spec.label.len() + v.spec.task_summary.len() + v.spec.incarnation.len();
        bytes += v.children.len() * std::mem::size_of::<String>();
        bytes += v.logs.len() * std::mem::size_of::<state::SanitizedLogLine>();
    }
    bytes
}

fn run_interactive_tui() -> Result<(), Box<dyn std::error::Error>> {
    let mut state = load_synthetic_demo_state();

    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout());
    let mut terminal = Terminal::new(backend)?;

    let mut synthetic_seq: u64 = 500;
    let res = (|| -> Result<(), Box<dyn std::error::Error>> {
        loop {
            terminal.draw(|frame| {
                let area = frame.area();
                let _ = render_console(&state, area, frame.buffer_mut());
            })?;

            if event::poll(Duration::from_millis(120))? {
                if let Event::Key(key) = event::read()? {
                    if let Some(action) = action_from_key(key) {
                        state.apply_action(action);
                    }
                }
            } else {
                // Periodic synthetic heartbeat/log pulse that never shifts selection
                synthetic_seq += 1;
                let now = state.now_ms + 120;
                state.apply_feed(FeedEvent::AdvanceClock { now_ms: now });
                if synthetic_seq % 25 == 0 {
                    state.apply_feed(FeedEvent::AppendLog {
                        node_id: "svc.desk.loop".to_string(),
                        seq: synthetic_seq,
                        timestamp_ms: now,
                        raw_text: format!(
                            "[SYNTHETIC-PULSE] seq={synthetic_seq} background event (focus invariant)"
                        ),
                    });
                }
            }

            if state.should_quit {
                break;
            }
        }
        Ok(())
    })();

    let _ = disable_raw_mode();
    let _ = stdout().execute(LeaveAlternateScreen);
    res
}

#[cfg(test)]
mod tests {
    use super::feed::{build_synthetic_10k_state, load_synthetic_demo_state, synthetic_log_burst};
    use super::layout::{buffer_to_ascii_string, compute_layout_plan, render_console};
    use super::state::{
        action_from_json, action_from_key, action_from_voice, sanitize_log_line, ActionParseError,
        ConsoleAction, ConsoleState, FeedEvent, NodeHealth, NodeSpec, UnknownReason, VoiceIntent,
    };
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use ratatui::buffer::Buffer;
    use ratatui::layout::Rect;

    #[test]
    fn zoom_back_empty_tree_and_deleted_recreated_node_preserve_identity() {
        // 1. Empty tree behavior: actions never panic and leave state valid
        let mut empty = ConsoleState::new(32);
        empty.apply_action(ConsoleAction::ZoomIn);
        empty.apply_action(ConsoleAction::Back);
        empty.apply_action(ConsoleAction::SelectNextVisible);
        assert_eq!(empty.selected_id, None);
        assert_eq!(empty.zoom_root_id, None);

        // 2. Multi-depth zoom and back invariance across depths 0 -> 1 -> 2 -> 3 -> 2 -> 1 -> 0
        let mut state = load_synthetic_demo_state();
        state.apply_action(ConsoleAction::Select("plane.ixctl".to_string()));
        state.apply_action(ConsoleAction::ZoomIn);
        assert_eq!(state.zoom_root_id.as_deref(), Some("plane.ixctl"));
        assert_eq!(
            state.breadcrumbs(),
            vec!["root.factory".to_string(), "plane.ixctl".to_string()]
        );

        state.apply_action(ConsoleAction::Select("svc.desk.loop".to_string()));
        state.apply_action(ConsoleAction::ZoomIn);
        assert_eq!(state.zoom_root_id.as_deref(), Some("svc.desk.loop"));
        assert_eq!(
            state.breadcrumbs(),
            vec![
                "root.factory".to_string(),
                "plane.ixctl".to_string(),
                "svc.desk.loop".to_string()
            ]
        );

        state.apply_action(ConsoleAction::Select("run.desk.loop.step1".to_string()));
        state.apply_action(ConsoleAction::ZoomIn);
        assert_eq!(state.zoom_root_id.as_deref(), Some("run.desk.loop.step1"));
        assert_eq!(
            state.breadcrumbs(),
            vec![
                "root.factory".to_string(),
                "plane.ixctl".to_string(),
                "svc.desk.loop".to_string(),
                "run.desk.loop.step1".to_string()
            ]
        );

        // Return up the parent chain one level at a time
        state.apply_action(ConsoleAction::Back);
        assert_eq!(state.zoom_root_id.as_deref(), Some("svc.desk.loop"));
        state.apply_action(ConsoleAction::Back);
        assert_eq!(state.zoom_root_id.as_deref(), Some("plane.ixctl"));
        state.apply_action(ConsoleAction::Back);
        assert_eq!(state.zoom_root_id.as_deref(), Some("root.factory"));

        // 3. Deleted and recreated node preserves selection and records incarnation change
        state.apply_action(ConsoleAction::Select("svc.desk.loop".to_string()));
        state.apply_feed(FeedEvent::RemoveNode {
            id: "svc.desk.loop".to_string(),
            timestamp_ms: 50_000,
        });
        assert_eq!(state.selected_id.as_deref(), Some("svc.desk.loop"));
        let removed_entry = state.nodes.get("svc.desk.loop").expect("tombstone kept");
        assert_eq!(removed_entry.removed_at_ms, Some(50_000));

        // Recreate with a new incarnation
        state.apply_feed(FeedEvent::UpsertNode(NodeSpec {
            id: "svc.desk.loop".to_string(),
            parent_id: Some("plane.ixctl".to_string()),
            label: "desk/loop-worker".to_string(),
            task_summary: "[SYNTHETIC] Recreated worker run".to_string(),
            incarnation: "inc-20261031-02".to_string(),
            reported_health: NodeHealth::Working,
            last_heartbeat_ms: Some(51_000),
            stale_after_ms: 5_000,
            progress_bp: Some(2500),
            quality_bp: Some(9100),
            evidence: vec![],
        }));
        let recreated = state.nodes.get("svc.desk.loop").expect("recreated node");
        assert_eq!(recreated.removed_at_ms, None);
        assert_eq!(
            recreated.previous_incarnation.as_deref(),
            Some("inc-20261031-01")
        );
        assert_eq!(recreated.incarnation_transitions, 1);
        assert_eq!(state.selected_id.as_deref(), Some("svc.desk.loop"));
    }

    #[test]
    fn selection_and_context_remain_stable_during_log_burst() {
        let mut state = load_synthetic_demo_state();
        state.max_logs_per_node = 32;
        state.apply_action(ConsoleAction::Select("svc.desk.loop".to_string()));
        state.apply_action(ConsoleAction::ZoomTo("plane.ixctl".to_string()));

        // Burst of logs on ANOTHER node and on the selected node
        for ev in synthetic_log_burst("probe.mux.root", 200, 1) {
            state.apply_feed(ev);
        }
        for ev in synthetic_log_burst("svc.desk.loop", 200, 1000) {
            state.apply_feed(ev);
        }

        // Neither selection nor zoom root nor reading scroll offset moved
        assert_eq!(state.selected_id.as_deref(), Some("svc.desk.loop"));
        assert_eq!(state.zoom_root_id.as_deref(), Some("plane.ixctl"));

        let entry = state.nodes.get("svc.desk.loop").expect("selected node");
        assert_eq!(entry.logs.len(), 32);
        assert!(entry.evicted_log_count >= 168);
        assert_eq!(entry.log_scroll_offset, 0);
    }

    #[test]
    fn resize_and_very_narrow_width_never_panic_and_render_cleanly() {
        let state = load_synthetic_demo_state();
        for (w, h) in [(0, 0), (8, 3), (16, 5), (28, 8), (60, 18), (110, 30)] {
            let area = Rect::new(0, 0, w, h);
            let mut buf = Buffer::empty(area);
            let plan = render_console(&state, area, &mut buf);
            let ascii = buffer_to_ascii_string(&buf);
            if w > 0 && h > 0 {
                assert!(!ascii.is_empty());
            }
            if w >= 80 && h >= 20 {
                assert!(!plan.degraded_narrow);
                assert!(!plan.icicle_cells.is_empty());
                assert!(ascii.contains("SYNTHETIC DEMO FEED"));
                assert!(ascii.contains("BREADCRUMB"));
            }
        }
    }

    #[test]
    fn stale_unknown_silence_not_idle_and_dangerous_chars_escaped() {
        let mut state = load_synthetic_demo_state();
        // Advance clock without heartbeats: Working/Idle nodes with finite stale_after_ms become Stale, never Idle
        state.apply_feed(FeedEvent::AdvanceClock { now_ms: 120_000 });
        let eff = state.effective_health("svc.desk.loop");
        assert!(
            matches!(eff, NodeHealth::Stale { .. }),
            "expected Stale on silent node, got {:?}",
            eff
        );

        // Gap detection transitions node to Unknown(GapDetected), not Idle
        state.apply_feed(FeedEvent::RecordGap {
            node_id: "probe.mux.root".to_string(),
            missed_seq: 42,
            timestamp_ms: 120_100,
        });
        assert_eq!(
            state.effective_health("probe.mux.root"),
            NodeHealth::Unknown(UnknownReason::GapDetected { missed_seq: 42 })
        );

        let mismatch = UnknownReason::EpochMismatch {
            expected: "e7".to_string(),
            observed: "e8".to_string(),
        };
        assert_eq!(mismatch.short_code(), "EPOCH(e7!=e8)");
        assert_eq!(NodeHealth::Failed.badge(), "FAIL");

        let cov = state.coverage_summary();
        assert!(cov.stale_nodes >= 1);
        assert!(cov.unknown_nodes >= 1);
        assert!(cov.gaps_detected >= 1);

        // Dangerous ANSI CSI, OSC, CR, NUL characters are escaped
        let raw = "normal \x1b[31mRED\x1b[0m \x1b]0;evil-title\x07 overwrite\rattack\x00end";
        let (clean, escaped) = sanitize_log_line(raw);
        assert!(escaped);
        assert!(!clean.contains('\x1b'));
        assert!(!clean.contains('\x07'));
        assert!(!clean.contains('\r'));
        assert!(!clean.contains('\x00'));
        assert!(clean.contains("^["));
        assert!(clean.contains("\\r"));
        assert!(clean.contains("\\0"));
    }

    #[test]
    fn keyboard_voice_and_json_actions_are_strictly_equivalent() {
        let key_zoom = action_from_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
            .expect("Enter maps to ZoomIn");
        let voice_zoom = action_from_voice(&VoiceIntent::ZoomBranch).expect("voice ZoomBranch");
        let json_zoom =
            action_from_json(r#"{"action":"zoom_in"}"#).expect("in-memory JSON zoom_in");
        assert_eq!(key_zoom, ConsoleAction::ZoomIn);
        assert_eq!(key_zoom, voice_zoom);
        assert_eq!(key_zoom, json_zoom);

        let key_back = action_from_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE))
            .expect("Backspace maps to Back");
        let voice_back = action_from_voice(&VoiceIntent::GoParent).expect("voice GoParent");
        let json_back = action_from_json(r#"{"action":"back"}"#).expect("in-memory JSON back");
        assert_eq!(key_back, ConsoleAction::Back);
        assert_eq!(key_back, voice_back);
        assert_eq!(key_back, json_back);

        // Apply each to 3 identical states and verify identical resulting state & layout
        let mut s_key = load_synthetic_demo_state();
        let mut s_voice = load_synthetic_demo_state();
        let mut s_json = load_synthetic_demo_state();

        let sel_voice = action_from_voice(&VoiceIntent::SelectNode {
            target_id: "plane.ixctl".to_string(),
        })
        .unwrap();
        let sel_json = action_from_json(r#"{"action":"select","node_id":"plane.ixctl"}"#).unwrap();
        s_key.apply_action(ConsoleAction::Select("plane.ixctl".to_string()));
        s_voice.apply_action(sel_voice);
        s_json.apply_action(sel_json);

        s_key.apply_action(key_zoom);
        s_voice.apply_action(voice_zoom);
        s_json.apply_action(json_zoom);

        assert_eq!(s_key.zoom_root_id, s_voice.zoom_root_id);
        assert_eq!(s_key.zoom_root_id, s_json.zoom_root_id);
        assert_eq!(s_key.breadcrumbs(), s_voice.breadcrumbs());
        assert_eq!(s_key.breadcrumbs(), s_json.breadcrumbs());

        let area = Rect::new(0, 0, 100, 28);
        let p_key = compute_layout_plan(&s_key, area);
        let p_voice = compute_layout_plan(&s_voice, area);
        let p_json = compute_layout_plan(&s_json, area);
        assert_eq!(p_key.icicle_cells, p_voice.icicle_cells);
        assert_eq!(p_key.icicle_cells, p_json.icicle_cells);

        // Exercise remaining action variants across all 3 input adapters
        assert_eq!(
            action_from_key(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE)),
            Some(ConsoleAction::ResetZoom)
        );
        assert_eq!(
            action_from_voice(&VoiceIntent::ResetRoot),
            Ok(ConsoleAction::ResetZoom)
        );
        assert_eq!(
            action_from_json(r#"{"action":"reset_zoom"}"#),
            Ok(ConsoleAction::ResetZoom)
        );

        assert_eq!(
            action_from_key(KeyEvent::new(KeyCode::Char('f'), KeyModifiers::NONE)),
            Some(ConsoleAction::ToggleFollowLogs)
        );
        assert_eq!(
            action_from_voice(&VoiceIntent::FollowLogs),
            Ok(ConsoleAction::ToggleFollowLogs)
        );
        assert_eq!(
            action_from_json(r#"{"action":"toggle_follow"}"#),
            Ok(ConsoleAction::ToggleFollowLogs)
        );

        assert_eq!(
            action_from_voice(&VoiceIntent::NextNode),
            Ok(ConsoleAction::SelectNextVisible)
        );
        assert_eq!(
            action_from_voice(&VoiceIntent::PreviousNode),
            Ok(ConsoleAction::SelectPrevVisible)
        );
        assert_eq!(
            action_from_voice(&VoiceIntent::ZoomToNode {
                target_id: "plane.ixctl".to_string()
            }),
            Ok(ConsoleAction::ZoomTo("plane.ixctl".to_string()))
        );
        assert_eq!(
            action_from_voice(&VoiceIntent::ScrollLogs { delta: -3 }),
            Ok(ConsoleAction::ScrollLogs(-3))
        );
        assert_eq!(
            action_from_voice(&VoiceIntent::TickClock { now_ms: 77_000 }),
            Ok(ConsoleAction::Tick { now_ms: 77_000 })
        );
        assert_eq!(
            action_from_voice(&VoiceIntent::QuitConsole),
            Ok(ConsoleAction::Quit)
        );
        assert_eq!(
            action_from_json(r#"{"action":"scroll_logs","delta":4}"#),
            Ok(ConsoleAction::ScrollLogs(4))
        );
        assert_eq!(
            action_from_json(r#"{"action":"tick","now_ms":77000}"#),
            Ok(ConsoleAction::Tick { now_ms: 77_000 })
        );
        assert_eq!(
            action_from_json(r#"{"action":"quit"}"#),
            Ok(ConsoleAction::Quit)
        );
        assert!(matches!(
            action_from_json("{bad"),
            Err(ActionParseError::InvalidJson(_))
        ));
        assert_eq!(
            action_from_json(r#"{"action":"select"}"#),
            Err(ActionParseError::MissingField("node_id"))
        );
        assert_eq!(
            action_from_json(r#"{"action":"nope"}"#),
            Err(ActionParseError::UnknownCommand("nope".to_string()))
        );

        s_key.apply_action(ConsoleAction::ResetZoom);
        s_key.apply_action(ConsoleAction::ToggleFollowLogs);
        s_key.apply_action(ConsoleAction::ScrollLogs(2));
        s_key.apply_action(ConsoleAction::ScrollLogs(-1));
        s_key.apply_action(ConsoleAction::Tick { now_ms: 80_000 });
        s_key.apply_action(ConsoleAction::Quit);
        assert!(s_key.should_quit);
    }

    #[test]
    fn depth_progress_quality_and_visual_span_are_orthogonal_and_10k_fixture_scales() {
        let mut state = build_synthetic_10k_state(10_000);
        assert_eq!(state.nodes.len(), 10_000);

        let area = Rect::new(0, 0, 110, 30);
        let plan_before = compute_layout_plan(&state, area);

        // Mutating progress_bp and quality_bp on `dom.00` must NEVER alter its depth or its visual span in the icicle
        let mut updated = state.nodes.get("dom.00").unwrap().spec.clone();
        updated.progress_bp = Some(100);
        updated.quality_bp = Some(9999);
        state.apply_feed(FeedEvent::UpsertNode(updated));

        let plan_after = compute_layout_plan(&state, area);
        assert_eq!(plan_before.icicle_cells, plan_after.icicle_cells);
    }

    #[test]
    fn the_two_panes_split_the_width_in_half() {
        let state = load_synthetic_demo_state();
        for width in [80u16, 108, 120, 161, 200] {
            let plan = compute_layout_plan(&state, Rect::new(0, 0, width, 30));
            let (l, r) = (plan.left_area.width, plan.right_area.width);
            assert_eq!(width, l + r, "width {width}");
            assert!(l.abs_diff(r) <= 1, "width {width}: left {l}, right {r}");
        }
    }

    /// The columns the icicle draws, by relative depth: `(rel_depth, col_start, col_width)`.
    fn columns(state: &ConsoleState, area: Rect) -> Vec<(usize, u16, u16)> {
        let mut cols: Vec<(usize, u16, u16)> = compute_layout_plan(state, area)
            .icicle_cells
            .iter()
            .map(|c| (c.rel_depth, c.col_start, c.col_width))
            .collect();
        cols.sort();
        cols.dedup();
        cols
    }

    #[test]
    fn the_levels_shown_share_the_whole_left_pane() {
        let area = Rect::new(0, 0, 160, 16);
        let mut state = load_synthetic_demo_state();
        // Three levels under plane.ixctl, four under the root: no empty column either way.
        for (zoom, levels) in [("plane.ixctl", 3usize), ("root.factory", 4)] {
            state.apply_action(ConsoleAction::ZoomTo(zoom.to_string()));
            let cols = columns(&state, area);
            assert_eq!(levels, cols.len(), "{zoom}: {cols:?}");
            let widths: Vec<u16> = cols.iter().map(|c| c.2).collect();
            let (min, max) = (*widths.iter().min().unwrap(), *widths.iter().max().unwrap());
            assert!(max - min <= 1, "{zoom}: {widths:?}");
            for pair in cols.windows(2) {
                assert_eq!(
                    pair[0].1 + pair[0].2,
                    pair[1].1,
                    "{zoom}: gap or overlap {cols:?}"
                );
            }
            let left = compute_layout_plan(&state, area).left_area;
            let (_, start, width) = cols[cols.len() - 1];
            assert_eq!(left.x + left.width - 1, start + width, "{zoom}: {cols:?}");
        }
    }

    #[test]
    fn arrows_walk_the_tree_and_zoom_only_to_keep_the_selection_in_view() {
        let mut state = load_synthetic_demo_state();
        state.apply_action(ConsoleAction::ZoomTo("root.factory".to_string()));
        state.apply_action(ConsoleAction::Select("svc.desk.loop".to_string()));
        let sel = |s: &ConsoleState| s.selected_id.clone().unwrap();
        let zoom = |s: &ConsoleState| s.zoom_root_id.clone().unwrap();

        state.apply_action(ConsoleAction::SelectChild);
        assert_eq!(
            (
                "run.desk.loop.step1".to_string(),
                "root.factory".to_string()
            ),
            (sel(&state), zoom(&state))
        );
        // A leaf has no child: the selection stays.
        state.apply_action(ConsoleAction::SelectChild);
        assert_eq!("run.desk.loop.step1", sel(&state));
        state.apply_action(ConsoleAction::SelectParent);
        state.apply_action(ConsoleAction::SelectParent);
        assert_eq!(
            ("plane.ixctl".to_string(), "root.factory".to_string()),
            (sel(&state), zoom(&state))
        );
        state.apply_action(ConsoleAction::SelectParent);
        state.apply_action(ConsoleAction::SelectParent);
        assert_eq!(
            ("root.factory".to_string(), "root.factory".to_string()),
            (sel(&state), zoom(&state))
        );

        // Leaving the zoom root upward zooms out with it.
        state.apply_action(ConsoleAction::ZoomTo("plane.ixctl".to_string()));
        state.apply_action(ConsoleAction::Select("plane.ixctl".to_string()));
        state.apply_action(ConsoleAction::SelectParent);
        assert_eq!(
            ("root.factory".to_string(), "root.factory".to_string()),
            (sel(&state), zoom(&state))
        );

        // A child past the four levels shown brings the zoom down to its parent.
        let mut deep = state.nodes["run.desk.loop.step1"].spec.clone();
        deep.id = "run.desk.loop.step1.sub".to_string();
        deep.parent_id = Some("run.desk.loop.step1".to_string());
        state.apply_feed(FeedEvent::UpsertNode(deep));
        state.apply_action(ConsoleAction::Select("run.desk.loop.step1".to_string()));
        state.apply_action(ConsoleAction::SelectChild);
        assert_eq!(
            (
                "run.desk.loop.step1.sub".to_string(),
                "run.desk.loop.step1".to_string()
            ),
            (sel(&state), zoom(&state))
        );
    }

    #[test]
    fn arrow_keys_select_while_enter_and_backspace_zoom() {
        let key = |code| action_from_key(KeyEvent::new(code, KeyModifiers::NONE));
        assert_eq!(Some(ConsoleAction::SelectChild), key(KeyCode::Right));
        assert_eq!(Some(ConsoleAction::SelectChild), key(KeyCode::Char('l')));
        assert_eq!(Some(ConsoleAction::SelectParent), key(KeyCode::Left));
        assert_eq!(Some(ConsoleAction::SelectParent), key(KeyCode::Char('h')));
        assert_eq!(Some(ConsoleAction::ZoomIn), key(KeyCode::Enter));
        assert_eq!(Some(ConsoleAction::Back), key(KeyCode::Backspace));
        assert_eq!(
            Ok(ConsoleAction::SelectChild),
            action_from_json(r#"{"action":"select_child"}"#)
        );
        assert_eq!(
            Ok(ConsoleAction::SelectParent),
            action_from_json(r#"{"action":"select_parent"}"#)
        );
        assert_eq!(
            Ok(ConsoleAction::SelectChild),
            action_from_voice(&VoiceIntent::ChildNode)
        );
        assert_eq!(
            Ok(ConsoleAction::SelectParent),
            action_from_voice(&VoiceIntent::ParentNode)
        );

        let state = load_synthetic_demo_state();
        let area = Rect::new(0, 0, 160, 16);
        let mut buf = Buffer::empty(area);
        render_console(&state, area, &mut buf);
        let screen = buffer_to_ascii_string(&buf);
        assert!(screen.contains("Right/l=Child"), "{screen}");
        assert!(screen.contains("Left/h=Parent"), "{screen}");
    }
}
