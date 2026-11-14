//! Seeded fractal-vs-Gaussian operator benchmark (refs #204, gap-matrix C5/C6).
//!
//! ```text
//! cargo run --release -p ix-evolution --example fractal_operator_bench
//! cargo run --release -p ix-evolution --example fractal_operator_bench -- --csv out.csv
//! ```
//!
//! Every arm is run on the same seeds with the same population size, generation
//! count, mutation rate and bounds. Only the operator differs, and every arm is
//! scaled to the same per-gene step size, so a difference here is a difference
//! in noise *structure* rather than in step length.
//!
//! The two `sched_ctrl_*` arms exist because the first run of this benchmark
//! could not tell a fractal result from an ordinary one. `T(0) = T(1) = 0`, so a
//! Takagi-modulated amplitude decays toward zero at the end of a run — that is
//! annealing, and annealing is expected to beat a constant step size on its own.
//! Both controls are matched to the Takagi schedule's *measured* mean multiplier,
//! so `takagi_schedule` only earns a fractal reading if it beats them too.
//!
//! Results are compared **paired within a seed**, never as two aggregate
//! medians. Run-to-run variance across seeds is larger than the effect being
//! looked for, so an unpaired comparison would mostly measure the seeds.

use ix_evolution::bench::{median, paired_compare, suite, Objective, PairedResult};
use ix_evolution::fractal::{AmplitudeSchedule, TakagiMode, TakagiNoise};
use ix_evolution::genetic::{CrossoverOperator, GeneticAlgorithm, MutationOperator};

/// Seeds per (objective, arm). 30 is the smallest count at which a clean sweep
/// is unambiguous under a sign test (p = 2^-29).
const SEEDS: u64 = 30;
const DIM: usize = 10;
const POPULATION: usize = 60;
const GENERATIONS: usize = 300;
const MUTATION_RATE: f64 = 0.2;
/// Swing of the Takagi amplitude schedule, and of the controls matched to it.
const SCHEDULE_STRENGTH: f64 = 0.8;

struct Arm {
    name: &'static str,
    mutation: MutationOperator,
    crossover: CrossoverOperator,
}

fn arms() -> Vec<Arm> {
    let noise = TakagiNoise::default();
    let takagi_schedule = AmplitudeSchedule::Takagi {
        noise,
        strength: SCHEDULE_STRENGTH,
    };
    // Measured, not nominal. The standardised curve has mean 0, so the nominal
    // multiplier is 1 — but its minimum is -3 standard deviations, which the
    // floor clips, pushing the real mean above 1. Matching the controls to the
    // nominal value would hand the fractal arm a larger average step and call
    // the resulting win structural.
    let mean = takagi_schedule.mean_multiplier(GENERATIONS);

    vec![
        Arm {
            name: "gaussian",
            mutation: MutationOperator::Gaussian,
            crossover: CrossoverOperator::Blx,
        },
        Arm {
            name: "takagi_iid",
            mutation: MutationOperator::Takagi {
                noise,
                mode: TakagiMode::Iid,
            },
            crossover: CrossoverOperator::Blx,
        },
        Arm {
            name: "takagi_correlated",
            mutation: MutationOperator::Takagi {
                noise,
                mode: TakagiMode::Correlated,
            },
            crossover: CrossoverOperator::Blx,
        },
        Arm {
            name: "takagi_schedule",
            mutation: MutationOperator::ScheduledGaussian(takagi_schedule),
            crossover: CrossoverOperator::Blx,
        },
        Arm {
            name: "sched_ctrl_constant",
            mutation: MutationOperator::ScheduledGaussian(AmplitudeSchedule::Constant(mean)),
            crossover: CrossoverOperator::Blx,
        },
        Arm {
            name: "sched_ctrl_linear",
            mutation: MutationOperator::ScheduledGaussian(AmplitudeSchedule::linear_matching(
                mean,
                GENERATIONS,
            )),
            crossover: CrossoverOperator::Blx,
        },
        Arm {
            name: "de_rham_crossover",
            mutation: MutationOperator::Gaussian,
            crossover: CrossoverOperator::DeRham {
                depth: 4,
                roughness: 0.3,
            },
        },
    ]
}

/// Best fitness per seed for one arm on one objective.
fn run_arm(arm: &Arm, objective: &Objective) -> Vec<f64> {
    (0..SEEDS)
        .map(|seed| {
            GeneticAlgorithm::new()
                .with_population_size(POPULATION)
                .with_generations(GENERATIONS)
                .with_mutation_rate(MUTATION_RATE)
                .with_bounds(objective.bounds.0, objective.bounds.1)
                .with_seed(seed)
                .with_mutation(arm.mutation)
                .with_crossover(arm.crossover)
                .minimize(&(objective.f), DIM)
                .best_fitness
        })
        .collect()
}

fn verdict(result: &PairedResult) -> &'static str {
    if result.p_value >= 0.05 {
        "no difference"
    } else if result.median_delta < 0.0 {
        "BETTER"
    } else {
        "worse"
    }
}

fn main() {
    let csv_path = {
        let mut args = std::env::args().skip(1);
        let mut path = None;
        while let Some(arg) = args.next() {
            if arg == "--csv" {
                path = args.next();
            }
        }
        path
    };

    let arms = arms();
    let mut csv = String::from("objective,arm,seed,best_fitness\n");
    let mut summary = String::new();

    println!(
        "fractal operator benchmark — dim={DIM} pop={POPULATION} gens={GENERATIONS} \
         rate={MUTATION_RATE} seeds=0..{}",
        SEEDS - 1
    );
    println!(
        "\n| objective | traits | arm | median best | vs gaussian (W/L/T) | median delta | p | verdict |"
    );
    println!("|---|---|---|---|---|---|---|---|");

    let mut head_to_head = String::new();

    for objective in suite() {
        let mut baseline: Vec<f64> = Vec::new();
        let mut by_arm: Vec<(&str, Vec<f64>)> = Vec::new();
        for (index, arm) in arms.iter().enumerate() {
            let results = run_arm(arm, &objective);
            by_arm.push((arm.name, results.clone()));
            for (seed, value) in results.iter().enumerate() {
                csv.push_str(&format!(
                    "{},{},{},{:.10}\n",
                    objective.name, arm.name, seed, value
                ));
            }
            if index == 0 {
                baseline = results.clone();
            }
            let mut sorted = results.clone();
            let med = median(&mut sorted);
            let traits = format!(
                "{}/{}",
                if objective.multimodal { "multi" } else { "uni" },
                if objective.coupled { "coupled" } else { "sep" }
            );

            if index == 0 {
                println!(
                    "| {} | {} | {} | {:.4} | — (baseline) | — | — | — |",
                    objective.name, traits, arm.name, med
                );
            } else {
                let cmp = paired_compare(&results, &baseline);
                println!(
                    "| {} | {} | {} | {:.4} | {}/{}/{} | {:+.4} | {:.4} | {} |",
                    objective.name,
                    traits,
                    arm.name,
                    med,
                    cmp.wins,
                    cmp.losses,
                    cmp.ties,
                    cmp.median_delta,
                    cmp.p_value,
                    verdict(&cmp)
                );
                summary.push_str(&format!(
                    "{} / {}: {}\n",
                    objective.name,
                    arm.name,
                    verdict(&cmp)
                ));
            }
        }

        // The comparison that decides whether the Takagi schedule is a fractal
        // result or an annealing one. Both controls have the same measured mean
        // amplitude, so this is scheduling *shape* against scheduling shape.
        let pick = |name: &str| -> Vec<f64> {
            by_arm
                .iter()
                .find(|(arm, _)| *arm == name)
                .map(|(_, values)| values.clone())
                .expect("arm present in the arm list")
        };
        let takagi = pick("takagi_schedule");
        for control in ["sched_ctrl_constant", "sched_ctrl_linear"] {
            let cmp = paired_compare(&takagi, &pick(control));
            head_to_head.push_str(&format!(
                "| {} | vs {} | {}/{}/{} | {:+.4} | {:.4} | {} |
",
                objective.name,
                control,
                cmp.wins,
                cmp.losses,
                cmp.ties,
                cmp.median_delta,
                cmp.p_value,
                verdict(&cmp)
            ));
        }
    }

    println!("
head-to-head: is the Takagi schedule fractal, or just annealing?");
    println!("(controls matched to its measured mean amplitude; verdict is for takagi_schedule)
");
    println!("| objective | comparison | W/L/T | median delta | p | verdict |");
    println!("|---|---|---|---|---|---|");
    print!("{head_to_head}");

    println!("\nverdict counts (p < 0.05, paired sign test):");
    for label in ["BETTER", "worse", "no difference"] {
        let count = summary
            .lines()
            .filter(|line| line.ends_with(label))
            .count();
        println!("  {label:<14} {count}");
    }

    if let Some(path) = csv_path {
        match std::fs::write(&path, &csv) {
            Ok(()) => println!("\nper-seed results written to {path}"),
            Err(error) => eprintln!("\nwriting {path}: {error}"),
        }
    }
}
