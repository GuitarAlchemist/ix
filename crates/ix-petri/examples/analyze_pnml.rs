//! Read a PNML Place/Transition net and print its behavioural report.
//!
//! ```text
//! cargo run -p ix-petri --example analyze_pnml -- path/to/net.pnml [max_states]
//! ```
//!
//! This is the fastest way to point the analyser at a net authored somewhere
//! else — which is the whole reason [`ix_petri::pnml`] reads the standard
//! format instead of a bespoke one.

use std::process::ExitCode;

use ix_petri::analysis::{analyze, Limits, Verdict};

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: analyze_pnml <file.pnml> [max_states]");
        return ExitCode::from(64);
    };
    let limits = match args.next() {
        Some(n) => match n.parse() {
            Ok(n) => Limits::with_max_states(n),
            Err(e) => {
                eprintln!("max_states: {e}");
                return ExitCode::from(64);
            }
        },
        None => Limits::default(),
    };

    let source = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let nets = match ix_petri::read_pnml(&source) {
        Ok(nets) => nets,
        Err(e) => {
            eprintln!("{path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut wedged = false;
    for pnml_net in &nets {
        let net = &pnml_net.net;
        let report = analyze(net, limits);
        println!(
            "net {} ({} places, {} transitions)",
            pnml_net.id,
            net.places().len(),
            net.transitions().len()
        );
        println!(
            "  states {}{}  edges {}",
            report.states,
            if report.truncated { " (truncated)" } else { "" },
            report.transitions_fired
        );
        match &report.deadlock_free {
            Verdict::Holds(_) => println!("  deadlock-free"),
            Verdict::Fails(deadlocks) => {
                wedged = true;
                println!("  DEADLOCK x{}", report.deadlock_count);
                for d in deadlocks {
                    println!("    marking: {}", d.marking);
                    println!(
                        "    witness: {}",
                        net.label_sequence(&d.witness).join(" -> ")
                    );
                }
            }
            Verdict::Unknown { reason } => println!("  deadlock: unknown ({reason})"),
        }
        match &report.bounded {
            Verdict::Holds(b) => println!("  {}-bounded", b.k),
            Verdict::Fails(_) => {
                let w = report
                    .unbounded_witness
                    .as_ref()
                    .expect("failure has a witness");
                println!(
                    "  UNBOUNDED: {} -> {} by [{}]",
                    w.smaller,
                    w.larger,
                    w.pumping_sequence.join(" -> ")
                );
            }
            Verdict::Unknown { reason } => println!("  boundedness: unknown ({reason})"),
        }
        report_ids("dead transitions", &report.quasi_live, net);
        report_ids("not live", &report.live, net);
        match &report.reversible {
            Verdict::Holds(()) => println!("  reversible"),
            Verdict::Fails(()) => println!("  not reversible"),
            Verdict::Unknown { .. } => println!("  reversibility: unknown"),
        }
    }

    if wedged {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn report_ids(label: &str, verdict: &Verdict<Vec<String>>, net: &ix_petri::PetriNet) {
    match verdict {
        Verdict::Holds(_) => {}
        Verdict::Fails(ids) => println!("  {label}: {}", net.label_sequence(ids).join(", ")),
        Verdict::Unknown { .. } => println!("  {label}: unknown"),
    }
}
