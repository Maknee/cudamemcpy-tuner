//! Offline libcuda table maintenance. Downloads are explicit; generation never
//! downloads, loads a driver, or patches a running process.

mod drivers;
mod image;
mod inline;
mod instructions;
mod scan;
mod tables;

use anyhow::{ensure, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use image::Arch;
use std::{
    collections::BTreeSet,
    io::{self, Write},
    path::PathBuf,
    process::ExitCode,
};

#[derive(Parser)]
#[command(
    version,
    about = "Download, scan and verify libcuda patch tables (x86-64 and AArch64)",
    after_help = "Typical workflow:\n  cmt-sigscan fetch\n  cmt-sigscan verify\n  cmt-sigscan generate\n\nNormal application builds use the checked-in tables. ARM scans are static checks, not hardware validation."
)]
struct Cli {
    /// Repository root containing src/gen and tools/sigscan
    #[arg(long, global = true, default_value = ".")]
    repo: PathBuf,
    /// Local driver cache (default: <repo>/tmp/drivers)
    #[arg(long, global = true)]
    drivers: Option<PathBuf>,
    /// Pinned package manifest (default: <repo>/tools/sigscan/drivers.manifest)
    #[arg(long, global = true)]
    manifest: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Args)]
struct Selection {
    /// Select one architecture; omitted means both
    #[arg(long)]
    arch: Option<Arch>,
    /// Filter listed/fetched versions, comma-separated (requires --arch)
    #[arg(long, value_delimiter = ',', requires = "arch")]
    versions: Vec<String>,
}

#[derive(Subcommand)]
enum Command {
    /// Show pinned versions and their expected local paths
    List {
        #[command(flatten)]
        selection: Selection,
    },
    /// Download verified packages and extract only libcuda; never run installers
    Fetch {
        #[command(flatten)]
        selection: Selection,
    },
    /// Inspect local ELF files and print a complete generated table or rows
    Scan {
        #[arg(required = true)]
        files: Vec<PathBuf>,
        /// Require this architecture for every input
        #[arg(long)]
        arch: Option<Arch>,
        #[arg(long, value_enum, default_value = "table")]
        format: Format,
        /// Show the instruction relationships used for each match
        #[arg(long)]
        explain: bool,
    },
    /// Check every manifest table entry; no files are changed
    #[command(visible_alias = "check")]
    Verify {
        /// Select one complete architecture table; omitted means both
        #[arg(long)]
        arch: Option<Arch>,
        #[arg(long)]
        explain: bool,
    },
    /// Regenerate every manifest table entry after all selected inputs pass
    Generate {
        /// Select one complete architecture table; omitted means both
        #[arg(long)]
        arch: Option<Arch>,
        #[arg(long)]
        explain: bool,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum Format {
    Table,
    Rows,
}

fn run(cli: Cli) -> Result<()> {
    let cache = cli.drivers.unwrap_or_else(|| cli.repo.join("tmp/drivers"));
    let manifest_path = cli
        .manifest
        .unwrap_or_else(|| cli.repo.join("tools/sigscan/drivers.manifest"));
    match cli.command {
        Command::Scan {
            files,
            arch,
            format,
            explain,
        } => {
            let mut rows = Vec::new();
            let mut versions = BTreeSet::new();
            let mut actual_arch = arch;
            for file in files {
                let scan = scan::file(&file, actual_arch, explain)?;
                actual_arch = Some(scan.arch);
                ensure!(
                    versions.insert(scan.version.clone()),
                    "duplicate driver version {}",
                    scan.version
                );
                rows.push(scan);
            }
            let output = match format {
                // A scan of local files has no manifest purpose; rows print as `table` rows.
                Format::Table => tables::render(
                    &rows.into_iter().map(|row| (row, false)).collect::<Vec<_>>(),
                )?,
                Format::Rows => rows
                    .iter()
                    .map(|row| tables::row(row, false))
                    .collect::<Result<Vec<_>>>()?
                    .concat(),
            };
            io::stdout()
                .lock()
                .write_all(output.as_bytes())
                .context("write generated output")?;
        }
        Command::List { selection } => {
            let manifest = drivers::manifest(&manifest_path)?;
            for driver in drivers::select(&manifest, selection.arch, &selection.versions)? {
                println!(
                    "{} {:<12} {:<6} {}",
                    driver.arch,
                    driver.version,
                    driver.purpose.name(),
                    driver.library(&cache).display()
                );
            }
        }
        Command::Fetch { selection } => {
            let manifest = drivers::manifest(&manifest_path)?;
            for driver in drivers::select(&manifest, selection.arch, &selection.versions)? {
                drivers::fetch(driver, &cache)
                    .with_context(|| format!("fetch {} {}", driver.arch, driver.version))?;
            }
        }
        Command::Verify { arch, explain } => {
            let manifest = drivers::manifest(&manifest_path)?;
            let updates = tables::prepare(&cli.repo, &cache, &manifest, arch, explain)?;
            for update in updates {
                ensure!(
                    update.original == update.generated,
                    "{} differs from scanner output; run generate and review the diff",
                    update.path.display()
                );
                println!("Verified {}", update.path.display());
            }
        }
        Command::Generate { arch, explain } => {
            let manifest = drivers::manifest(&manifest_path)?;
            let updates = tables::prepare(&cli.repo, &cache, &manifest, arch, explain)?;
            tables::publish(&updates)?;
            for update in updates {
                println!("Generated {}", update.path.display());
            }
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("cmt-sigscan: {error:#}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/fixtures/sigscan/rust.rs"]
mod tests;
