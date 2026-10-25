//! `hnswops` command-line interface.

mod cli;
mod demo;

use std::process::ExitCode;

use clap::Parser;

fn main() -> ExitCode {
    let filter = tracing_subscriber::EnvFilter::try_from_env("HNSWOPS_LOG")
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
        .init();
    let args = cli::Cli::parse();
    match cli::run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
