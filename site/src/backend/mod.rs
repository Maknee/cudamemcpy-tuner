//! Offline conversion, import and static export. Nothing here is deployed.
mod cli;
mod db;
mod model;
mod publish;
mod scrub;
mod tune;
pub use cli::run as cli;

fn db_path() -> String {
    std::env::var("CMX_DB").unwrap_or_else(|_| "data/cmx.duckdb".into())
}
