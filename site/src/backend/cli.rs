//! Offline conversion and database import commands.

use super::{db, db_path, publish, tune};

pub fn run(args: &[String]) -> i32 {
    match args.get(1).map(String::as_str) {
        Some("import") => cmd_import(&args[2..]),
        Some("convert") => tune::cmd_convert(&args[2..]),
        Some("build") => publish::command(&args[2..]),
        None | Some("help" | "-h" | "--help") => {
            eprintln!(
                "cudamemcpy-site convert <tune.json> --id ID --machine NAME [-o run.json]\n\
                cudamemcpy-site import <run.json>...\n\
                cudamemcpy-site build --web PATH --out PATH  export $CMX_DB as a static site"
            );
            0
        }
        Some(command) => {
            eprintln!("unknown command {command:?}; see cudamemcpy-site --help");
            2
        }
    }
}

fn cmd_import(files: &[String]) -> i32 {
    if files.is_empty() {
        eprintln!("import: provide at least one run.json file");
        return 2;
    }
    let path = db_path();
    if let Some(dir) = std::path::Path::new(&path).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let mut conn = match db::open(&path, false) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("open {path}: {e}");
            return 1;
        }
    };
    let mut bad = 0;
    for f in files {
        match std::fs::read(f).map_err(|e| e.to_string()).and_then(|b| db::import(&mut conn, &b)) {
            Ok((id, tier, n)) => println!("imported {id:<40} {n:>8} rows  verify={tier}"),
            Err(e) => {
                eprintln!("{f}: {e}");
                bad += 1;
            }
        }
    }
    let _ = conn.execute_batch("CHECKPOINT");
    if bad > 0 {
        1
    } else {
        0
    }
}
