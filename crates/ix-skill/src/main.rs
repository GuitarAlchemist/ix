//! ix — Claude Code ML skill CLI.
//!
//! Eight-verb noun-verb grammar:
//! `run | pipeline | list | describe | check | beliefs | demo | serve`.
//!
//! Global flags:
//! - `--format {auto,table,json,jsonl,yaml}` (default: auto — table on TTY, json on pipe)
//! - `--quiet` / `--verbose` — suppress / increase log verbosity
//! - `--no-color` — disable ANSI (also honors `NO_COLOR` env)

use clap::{Parser, Subcommand};

use ix_skill::exit;
use ix_skill::output::Format;
use ix_skill::verbs;

#[derive(Parser)]
#[command(
    name = "ix",
    version,
    about = "ML algorithms + governance CLI for the GuitarAlchemist ecosystem",
    long_about = None,
)]
struct Cli {
    /// Output format (default: auto — table on TTY, json on pipe)
    #[arg(long, short = 'f', value_enum, default_value_t = Format::Auto, global = true)]
    format: Format,

    /// Suppress progress output; errors still printed
    #[arg(long, short = 'q', global = true)]
    quiet: bool,

    /// Increase log verbosity (-v, -vv)
    #[arg(long, short = 'v', global = true, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Disable ANSI color output
    #[arg(long, global = true)]
    no_color: bool,

    #[command(subcommand)]
    command: Verb,
}

#[derive(Subcommand)]
enum Verb {
    /// Execute a single skill with JSON input
    Run {
        /// Dotted skill name (see `ix list skills`)
        skill: String,
        /// JSON input from file
        #[arg(long, short = 'i')]
        input_file: Option<String>,
        /// Inline JSON input literal
        #[arg(long = "input")]
        input_literal: Option<String>,
    },

    /// List capabilities from the registry and governance artifacts
    List {
        #[command(subcommand)]
        noun: ListNoun,
    },

    /// Show schema + metadata for a registered entity
    Describe {
        #[command(subcommand)]
        noun: DescribeNoun,
    },

    /// Validate the environment or check an action against the constitution
    Check {
        #[command(subcommand)]
        noun: CheckNoun,
    },

    /// Manage belief state files (`state/beliefs/*.belief.json`)
    Beliefs {
        #[command(subcommand)]
        noun: BeliefsNoun,
    },

    /// Visual DAG pipelines (stub — full impl in Week 4)
    Pipeline {
        #[command(subcommand)]
        noun: PipelineNoun,
    },

    /// Run curated demo scenarios showcasing ix capabilities
    Demo {
        #[command(subcommand)]
        noun: DemoNoun,
    },

    /// Long-running services (stub — MCP server, REPL, etc.)
    Serve {
        #[command(subcommand)]
        noun: ServeNoun,
    },

    /// Print the maturity-tier "Stable" crates with a public-API hash each.
    ///
    /// CI uses this on `main` and on PR branches and fails the PR if a
    /// Stable crate's hash changed without an explicit demotion. See
    /// `crate-maturity.toml` for the source of truth and the README's
    /// "Stability contract" section for the policy.
    StableSurface {
        /// Include non-Stable crates in the report (default: Stable only).
        #[arg(long)]
        all_tiers: bool,
    },
}

#[derive(Subcommand)]
enum DemoNoun {
    /// List all available demo scenarios
    List,
    /// Show scenario details and steps without executing
    Describe {
        /// Scenario id (e.g. "chaos-detective")
        scenario: String,
    },
    /// Execute a demo scenario end-to-end
    Run {
        /// Scenario id (e.g. "chaos-detective")
        scenario: String,
        /// RNG seed for reproducible data (default: 42)
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Verbosity: 0=terse, 1=normal, 2=verbose
        #[arg(long, short = 'V', default_value_t = 1)]
        verbosity: u8,
    },
}

#[derive(Subcommand)]
enum ListNoun {
    /// Every registered skill (optionally filtered by domain or query)
    Skills {
        #[arg(long)]
        domain: Option<String>,
        #[arg(long)]
        query: Option<String>,
        /// Include per-skill arg json_schema, governance_tags, and a
        /// pipeline-callable (arity-1) flag — the bulk catalog an NL→pipeline
        /// generator needs in one call.
        #[arg(long)]
        schemas: bool,
    },
    /// Distinct domains across registered skills, with counts
    Domains,
    /// Demerzel personas on disk
    Personas,
}

#[derive(Subcommand)]
enum DescribeNoun {
    /// Signature, schema and governance tags of one skill
    Skill { name: String },
    /// Load and show a Demerzel persona by name
    Persona { name: String },
    /// Load and show a policy from governance/demerzel/policies/
    Policy { name: String },
}

#[derive(Subcommand)]
enum CheckNoun {
    /// Environment / registry / governance health
    Doctor,
    /// Evaluate a proposed action against the constitution
    Action {
        /// Free-text description of the proposed action
        text: String,
        /// Optional context describing the operating environment
        #[arg(long)]
        context: Option<String>,
    },
}

#[derive(Subcommand)]
enum BeliefsNoun {
    /// List all belief files
    Show,
    /// Print one belief file by key
    Get { key: String },
    /// Write a new belief file
    Set {
        /// Belief key (used as filename stem)
        key: String,
        /// Proposition text
        proposition: String,
        /// Hexavalent truth value (T/P/U/D/F/C)
        #[arg(long, default_value = "U")]
        truth: String,
        /// Confidence 0.0–1.0
        #[arg(long, default_value_t = 0.5)]
        confidence: f64,
    },
    /// Capture all current beliefs into a timestamped snapshot file
    Snapshot {
        /// Short kebab-case description of why you took this snapshot
        description: String,
    },
}

#[derive(Subcommand)]
enum PipelineNoun {
    /// Scaffold a minimal `ix.yaml` with one stage
    New { name: String },
    /// Execute an ix.yaml pipeline
    Run {
        /// Path to the pipeline file (default: ./ix.yaml)
        #[arg(long)]
        file: Option<String>,
        /// Stream NDJSON events to stdout while the pipeline runs
        #[arg(long)]
        json: bool,
        /// Bind a run-time parameter referenced by `{param: "name"}` in the
        /// spec. Repeatable. VALUE is parsed as JSON, or `@path` reads JSON from
        /// a file; a bare unparseable word is taken as a string. Example:
        /// --param dataset=@data.json --param k=3
        #[arg(long = "param", value_name = "NAME=VALUE")]
        param: Vec<String>,
    },
    /// Show execution-level DAG structure
    Dag {
        #[arg(long)]
        file: Option<String>,
    },
    /// Parse + lower without executing (catches cycles, missing skills)
    Validate {
        #[arg(long)]
        file: Option<String>,
    },
    /// Emit the PipelineSpec JSON Schema (skill enum drawn from the registry)
    Schema,
    /// Compile a natural-language request into a pipeline via the LLM proposer,
    /// validate (lower), gate (governance), and optionally run + narrate.
    Compile {
        /// The natural-language request, e.g. "compute stats on some numbers then audit it".
        sentence: String,
        /// Max self-repair rounds when the generated spec fails validation.
        #[arg(long, default_value_t = 3)]
        max_rounds: u32,
        /// Execute the compiled pipeline (default: compile + gate only).
        #[arg(long)]
        run: bool,
        /// Bind a run-time parameter the proposer emitted for absent data
        /// ({param: "name"}). Repeatable; same syntax as `pipeline run --param`.
        /// Only meaningful with --run. e.g. --param dataset=@data.json
        #[arg(long = "param", value_name = "NAME=VALUE")]
        param: Vec<String>,
    },
    /// Aggregate the NL→pipeline translation ledger (hits.jsonl) into a yield
    /// metric paired with its refusal guardrails (Goodhart-resistant: a yield
    /// gain alongside a coverage-refusal drop is gate-loosening, not progress).
    Hits,
}

#[derive(Subcommand)]
enum ServeNoun {
    /// Print a coming-soon message
    Mcp,
    Repl,
}

fn main() {
    // Honor NO_COLOR env (CLI --no-color also disables, handled in output).
    if std::env::var_os("NO_COLOR").is_some() {
        // Nothing to configure globally at this point — per-output call sites
        // respect the env var where needed.
    }

    let cli = Cli::parse();
    let exit_code = dispatch(cli);
    std::process::exit(exit_code);
}

fn dispatch(cli: Cli) -> i32 {
    let fmt = cli.format;
    match cli.command {
        Verb::Run {
            skill,
            input_file,
            input_literal,
        } => match verbs::run::run(&skill, input_file.as_deref(), input_literal.as_deref(), fmt) {
            Ok(()) => exit::OK_TRUE,
            Err(e) => {
                eprintln!("ix run: {e}");
                exit::RUNTIME_ERROR
            }
        },

        Verb::List { noun } => match noun {
            ListNoun::Skills {
                domain,
                query,
                schemas,
            } => try_or(verbs::list::skills(
                domain.as_deref(),
                query.as_deref(),
                schemas,
                fmt,
            )),
            ListNoun::Domains => try_or(verbs::list::domains(fmt)),
            ListNoun::Personas => try_or(verbs::list::personas(fmt)),
        },

        Verb::Describe { noun } => match noun {
            DescribeNoun::Skill { name } => try_or(verbs::describe::skill(&name, fmt)),
            DescribeNoun::Persona { name } => try_or(verbs::describe::persona(&name, fmt)),
            DescribeNoun::Policy { name } => try_or(verbs::describe::policy(&name, fmt)),
        },

        Verb::Check { noun } => match noun {
            CheckNoun::Doctor => match verbs::check::doctor(fmt) {
                Ok(code) => code,
                Err(e) => {
                    eprintln!("ix check doctor: {e}");
                    exit::RUNTIME_ERROR
                }
            },
            CheckNoun::Action { text, context } => {
                match verbs::check::action(&text, context.as_deref(), fmt) {
                    Ok(code) => code,
                    Err(e) => {
                        eprintln!("ix check action: {e}");
                        exit::RUNTIME_ERROR
                    }
                }
            }
        },

        Verb::Beliefs { noun } => match noun {
            BeliefsNoun::Show => try_or(verbs::beliefs::show(fmt)),
            BeliefsNoun::Get { key } => try_or(verbs::beliefs::get(&key, fmt)),
            BeliefsNoun::Set {
                key,
                proposition,
                truth,
                confidence,
            } => try_or(verbs::beliefs::set(
                &key,
                &proposition,
                &truth,
                confidence,
                fmt,
            )),
            BeliefsNoun::Snapshot { description } => {
                try_or(verbs::beliefs::snapshot(&description, fmt))
            }
        },

        Verb::Pipeline { noun } => match noun {
            PipelineNoun::New { name } => try_or(verbs::pipeline::new(&name, fmt)),
            PipelineNoun::Validate { file } => {
                try_or(verbs::pipeline::validate(file.as_deref(), fmt))
            }
            PipelineNoun::Dag { file } => try_or(verbs::pipeline::dag(file.as_deref(), fmt)),
            PipelineNoun::Run { file, json, param } => {
                try_or(verbs::pipeline::run(file.as_deref(), json, &param, fmt))
            }
            PipelineNoun::Schema => try_or(verbs::pipeline::schema(fmt)),
            PipelineNoun::Compile {
                sentence,
                max_rounds,
                run,
                param,
            } => try_or(verbs::compile::compile(
                &sentence, max_rounds, run, &param, fmt,
            )),
            PipelineNoun::Hits => try_or(verbs::compile::hits(fmt)),
        },

        Verb::Demo { noun } => match noun {
            DemoNoun::List => try_or(verbs::demo::list(fmt)),
            DemoNoun::Describe { scenario } => try_or(verbs::demo::describe(&scenario, fmt)),
            DemoNoun::Run {
                scenario,
                seed,
                verbosity,
            } => try_or(verbs::demo::run(&scenario, seed, verbosity, fmt)),
        },

        Verb::Serve { noun } => {
            let msg = match noun {
                ServeNoun::Mcp => "ix serve mcp: run `ix-mcp` directly for now",
                ServeNoun::Repl => "ix serve repl: coming in a later phase",
            };
            eprintln!("{msg}");
            exit::UNKNOWN
        }

        Verb::StableSurface { all_tiers } => try_or(verbs::stable_surface::run(fmt, all_tiers)),
    }
}

fn try_or(r: Result<(), String>) -> i32 {
    match r {
        Ok(()) => exit::OK_TRUE,
        Err(e) => {
            eprintln!("ix: {e}");
            exit::RUNTIME_ERROR
        }
    }
}
