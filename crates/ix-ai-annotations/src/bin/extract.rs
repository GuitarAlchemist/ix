//! `ix-ai-annotations` CLI — extract every `@ai:` marker in the workspace and
//! write a JSONL stream to `state/quality/ai-annotations.jsonl`.

use clap::Parser;
use ix_ai_annotations::extract_workspace;
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "ix-ai-annotations",
    about = "Extract @ai: source-code annotations to a JSONL stream"
)]
struct Args {
    /// Workspace root to scan.
    #[arg(long, default_value = ".")]
    workspace: PathBuf,

    /// Output file. Default: <workspace>/state/quality/ai-annotations.jsonl
    #[arg(long)]
    out: Option<PathBuf>,

    /// Print a one-line summary to stderr.
    #[arg(long, default_value_t = false)]
    quiet: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let out_path = args.out.unwrap_or_else(|| {
        args.workspace
            .join("state")
            .join("quality")
            .join("ai-annotations.jsonl")
    });

    let annotations = extract_workspace(&args.workspace)?;

    if let Some(parent) = out_path.parent() {
        create_dir_all(parent)?;
    }
    let mut f = BufWriter::new(File::create(&out_path)?);
    for a in &annotations {
        serde_json::to_writer(&mut f, a)?;
        f.write_all(b"\n")?;
    }
    f.flush()?;

    if !args.quiet {
        eprintln!(
            "ix-ai-annotations: {} annotations -> {}",
            annotations.len(),
            out_path.display()
        );
    }
    Ok(())
}
