//! Build a bounded static snapshot; the previous output survives any failed build.
use super::{db, db_path, scrub};
use crate::data::{Detail, PackedPoint, PackedPoints, RunRow, Summary};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::{Path, PathBuf},
};

const MAX_FILE: u64 = 2_000_000;
const MAX_TOTAL: u64 = 8_000_000;

pub fn command(args: &[String]) -> i32 {
    let mut web = None;
    let mut out = None;
    for pair in args.chunks(2) {
        if pair.len() != 2 {
            eprintln!("build: --web PATH --out PATH are required");
            return 2;
        }
        match pair[0].as_str() {
            "--web" => web = Some(PathBuf::from(&pair[1])),
            "--out" => out = Some(PathBuf::from(&pair[1])),
            _ => {
                eprintln!("build: unknown option {}", pair[0]);
                return 2;
            }
        }
    }
    let (Some(web), Some(out)) = (web, out) else {
        eprintln!("build: --web PATH --out PATH are required");
        return 2;
    };
    match build(&web, &out) {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("build: {error}");
            1
        }
    }
}

fn write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    fs::create_dir_all(path.parent().ok_or("missing parent")?).map_err(|e| e.to_string())?;
    fs::write(path, bytes).map_err(|e| e.to_string())
}

fn json_file(path: &Path, value: &impl serde::Serialize) -> Result<(), String> {
    write(path, &serde_json::to_vec(value).map_err(|e| e.to_string())?)
}

fn copy_web(source: &Path, target: &Path) -> Result<(), String> {
    for entry in fs::read_dir(source).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let kind = entry.file_type().map_err(|e| e.to_string())?;
        let path = entry.path();
        let to = target.join(entry.file_name());
        if kind.is_dir() {
            copy_web(&path, &to)?;
        } else if kind.is_file() && !matches!(path.extension().and_then(|s| s.to_str()), Some("br" | "gz")) {
            write(&to, &fs::read(&path).map_err(|e| e.to_string())?)?;
        } else if kind.is_symlink() {
            return Err("web assets must not contain symlinks".into());
        }
    }
    Ok(())
}

fn sizes(directory: &Path, root: &Path, files: &mut BTreeMap<String, u64>) -> Result<(), String> {
    for entry in fs::read_dir(directory).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if path.is_dir() {
            sizes(&path, root, files)?;
        } else {
            let size = entry.metadata().map_err(|e| e.to_string())?.len();
            if size >= MAX_FILE {
                return Err(format!("{} is {size} bytes; each file must be under {MAX_FILE}", path.display()));
            }
            files.insert(path.strip_prefix(root).unwrap().to_string_lossy().into_owned(), size);
        }
    }
    Ok(())
}

fn index<T: Eq + std::hash::Hash + Clone>(values: &mut Vec<T>, ids: &mut HashMap<T, usize>, value: T) -> usize {
    *ids.entry(value.clone()).or_insert_with(|| {
        let i = values.len();
        values.push(value);
        i
    })
}

// The CLI reads the existing summary to select a run, then fetches this compact tune.
// Keep the tuner's selection-line ordering and exact byte count so its normal loader validates it.
fn remote_tune(run: &RunRow, selection: &Value) -> Option<Vec<u8>> {
    let mut selection = selection.clone();
    let tail = json!({
        "schema": "cmt.tuning", "version": 1, "state": "complete", "matrix_state": "complete",
        "selection_policy": selection.get("policy")?.as_str()?,
        "remote": {"source": "cudamemcpy.com", "run": run.id}, "runs": []
    }).to_string();
    let mut size = 0;
    loop {
        selection.as_object_mut()?.insert("file_bytes".into(), json!(size));
        let text = format!("{{\"selection\":{},\n{}\n", scrub::selection_json(&selection)?, &tail[1..]);
        if text.len() == size {
            return Some(text.into_bytes());
        }
        size = text.len();
    }
}

fn snapshot(web: &Path, out: &Path) -> Result<(), String> {
    let html = fs::read(web.join("index.html")).map_err(|e| format!("browser index.html: {e}"))?;
    copy_web(web, out)?;
    write(&out.join("404.html"), b"<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><title>Not found</title><h1>Page not found</h1><a href=\"/\">Browse benchmarks</a></html>")?;
    let conn = db::open(&db_path(), true).map_err(|e| e.to_string())?;
    let runs = db::load_runs(&conn).map_err(|e| e.to_string())?;
    let curves = db::load_curves(&conn).map_err(|e| e.to_string())?;
    json_file(&out.join("data/summary.json"), &Summary { runs: runs.clone(), curves: curves.clone() })?;
    for run in &runs {
        if run.id.is_empty()
            || run.id == "."
            || run.id == ".."
            || !run.id.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
        {
            return Err(format!("unsafe run ID {:?}", run.id));
        }
        let dir = out.join("data").join(&run.id);
        write(&out.join("run").join(&run.id).join("index.html"), &html)?;
        let download = db::export_meta(&conn, run).map_err(|e| e.to_string())?;
        json_file(&dir.join("download.json"), &download)?;
        if run.has_selection {
            let tune = remote_tune(run, &download["selection"])
                .ok_or_else(|| format!("invalid selection for {}", run.id))?;
            write(&dir.join("tune.json"), &tune)?;
        }
        let meta = db::run_meta(&conn, &run.id).map_err(|e| e.to_string())?;
        let meta = meta
            .as_object()
            .map(|m| {
                m.iter()
                    .map(|(k, v)| (k.clone(), v.as_str().map(str::to_owned).unwrap_or_else(|| v.to_string())))
                    .collect()
            })
            .unwrap_or_default();
        json_file(
            &dir.join("detail.json"),
            &Detail {
                run: run.clone(),
                meta,
                curves: curves.get(&run.id).cloned().unwrap_or_default(),
                top: db::top_points(&conn, &run.id, 3).map_err(|e| e.to_string())?,
            },
        )?;
        let mut packed = PackedPoints { dirs: vec![], sizes: vec![], configs: vec![], files: vec![] };
        let (mut dirs, mut sizes, mut configs) = (HashMap::new(), HashMap::new(), HashMap::new());
        let mut statement = conn.prepare("SELECT dir,size,config,gbps,n,cv,baseline,selected FROM points WHERE run_id=? ORDER BY dir,size,config").map_err(|e| e.to_string())?;
        let rows = statement
            .query_map([&run.id], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, f64>(3)?,
                    r.get::<_, Option<i32>>(4)?,
                    r.get::<_, Option<f64>>(5)?,
                    r.get::<_, bool>(6)?,
                    r.get::<_, bool>(7)?,
                ))
            })
            .map_err(|e| e.to_string())?;
        let mut chunk = Vec::new();
        for row in rows {
            let (d, s, c, g, n, cv, b, selected) = row.map_err(|e| e.to_string())?;
            chunk.push(PackedPoint(
                index(&mut packed.dirs, &mut dirs, d),
                index(&mut packed.sizes, &mut sizes, s),
                index(&mut packed.configs, &mut configs, c),
                g,
                n,
                cv,
                u8::from(b) | (u8::from(selected) << 1),
            ));
            if chunk.len() == 10_000 {
                let name = format!("points-{}.json", packed.files.len());
                json_file(&dir.join(&name), &chunk)?;
                packed.files.push(name);
                chunk.clear();
            }
        }
        if !chunk.is_empty() {
            let name = format!("points-{}.json", packed.files.len());
            json_file(&dir.join(&name), &chunk)?;
            packed.files.push(name);
        }
        json_file(&dir.join("points.json"), &packed)?;
    }
    Ok(())
}

fn build(web: &Path, out: &Path) -> Result<(), String> {
    let parent = out.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let source = fs::canonicalize(web).map_err(|e| e.to_string())?;
    let destination = if out.exists() {
        fs::canonicalize(out).map_err(|e| e.to_string())?
    } else {
        fs::canonicalize(parent).map_err(|e| e.to_string())?.join(out.file_name().ok_or("invalid output path")?)
    };
    if source.starts_with(&destination) || destination.starts_with(&source) {
        return Err("browser source and output directories must not overlap".into());
    }
    let next = parent.join(format!(".static-build-{}", std::process::id()));
    fs::create_dir(&next).map_err(|e| e.to_string())?;
    let result = (|| {
        snapshot(web, &next)?;
        let mut files = BTreeMap::new();
        sizes(&next, &next, &mut files)?;
        // Report counts too; measure again after writing it so the report is in the budget.
        json_file(
            &next.join("sizes.json"),
            &json!({"max_file_bytes":MAX_FILE,"max_total_bytes":MAX_TOTAL,"files":files}),
        )?;
        files.clear();
        sizes(&next, &next, &mut files)?;
        let total: u64 = files.values().sum();
        if total >= MAX_TOTAL {
            return Err(format!("static site is {total} bytes; must be under {MAX_TOTAL}"));
        }
        let old = parent.join(format!(".static-old-{}", std::process::id()));
        if old.exists() {
            return Err(format!("remove stale backup {} before building", old.display()));
        }
        if out.exists() {
            fs::rename(out, &old).map_err(|e| e.to_string())?;
        }
        if let Err(e) = fs::rename(&next, out) {
            if old.exists() {
                let _ = fs::rename(&old, out);
            }
            return Err(e.to_string());
        }
        if old.exists() {
            fs::remove_dir_all(old).map_err(|e| e.to_string())?;
        }
        println!(
            "Static site: {} files, {total} bytes total, largest {} bytes → {}",
            files.len(),
            files.values().max().copied().unwrap_or(0),
            out.display()
        );
        Ok(())
    })();
    if next.exists() {
        let _ = fs::remove_dir_all(next);
    }
    result
}
