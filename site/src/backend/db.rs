//! DuckDB storage: schema, JSON import, and the read-side cache built at boot.

use crate::backend::{
    model::{self, RunFile},
    scrub,
};
use duckdb::{params, Connection};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS runs (
  id           VARCHAR PRIMARY KEY,
  machine      VARCHAR NOT NULL,
  gpu          VARCHAR,
  cpu          VARCHAR,
  tier         VARCHAR NOT NULL,
  status       VARCHAR NOT NULL,
  note         VARCHAR,
  pcie_gen     UTINYINT,
  pcie_width   UTINYINT,
  link_gbps    DOUBLE,
  started_at   VARCHAR,
  meta         VARCHAR,          -- JSON object of string -> string
  verify_tier  VARCHAR NOT NULL, -- T0 | T1 | T2 | T3 | NC
  verify_notes VARCHAR,          -- JSON array of strings
  machine_json VARCHAR,          -- CPU model + arch, GPU name, driver version (scrub::public_machine)
  selection    VARCHAR,          -- tuner selection object (JSON) or NULL
  source_sha256 VARCHAR,
  imported_at  TIMESTAMP DEFAULT current_timestamp
);
CREATE TABLE IF NOT EXISTS points (
  run_id   VARCHAR NOT NULL,
  dir      VARCHAR NOT NULL,
  size     BIGINT  NOT NULL,
  config   VARCHAR NOT NULL,
  gbps     DOUBLE  NOT NULL,
  n        INTEGER,
  cv       DOUBLE,
  baseline BOOLEAN NOT NULL,
  selected BOOLEAN NOT NULL DEFAULT false
);
"#;

pub fn open(path: &str, read_only: bool) -> duckdb::Result<Connection> {
    let cfg = duckdb::Config::default()
        .access_mode(if read_only { duckdb::AccessMode::ReadOnly } else { duckdb::AccessMode::ReadWrite })?
        .threads(2)?;
    let c = Connection::open_with_flags(path, cfg)?;
    if !read_only {
        c.execute_batch(SCHEMA)?;
    }
    Ok(c)
}

/// Import one `cudamemcpy.run/v1` file. Replaces any existing run with the same id.
pub fn import(conn: &mut Connection, bytes: &[u8]) -> Result<(String, &'static str, usize), String> {
    // scrub again on import, so a hand-made or old run file can't publish identifying data
    let mut doc: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| format!("parse: {e}"))?;
    scrub::scrub(&mut doc);
    let mut f: RunFile = serde_json::from_value(doc).map_err(|e| format!("parse: {e}"))?;
    // and keep only what we publish of the machine, selection and meta
    f.run.meta.retain(|k, _| scrub::public_meta_key(k));
    f.machine = f.machine.as_ref().map(scrub::public_machine).filter(|m| !m.is_null());
    f.selection = f.selection.as_ref().map(scrub::public_selection).filter(|s| !s.is_null());
    let v = model::verify(&f)?;
    // sha256 of the file as stored (sorted keys, as before)
    let clean = serde_json::to_value(&f).and_then(|v| serde_json::to_vec(&v)).map_err(|e| e.to_string())?;
    let sha = hex(&Sha256::digest(&clean));
    let r = &f.run;
    let tx = conn.transaction().map_err(|e| e.to_string())?;
    tx.execute("DELETE FROM points WHERE run_id = ?", params![r.id]).map_err(|e| e.to_string())?;
    tx.execute("DELETE FROM runs WHERE id = ?", params![r.id]).map_err(|e| e.to_string())?;
    tx.execute(
        "INSERT INTO runs (id,machine,gpu,cpu,tier,status,note,pcie_gen,pcie_width,link_gbps,started_at,meta,verify_tier,verify_notes,source_sha256,machine_json,selection)
         VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        params![
            r.id, r.machine, r.gpu, r.cpu, r.tier, r.status, r.note,
            r.pcie_gen, r.pcie_width, model::link_gbps(r.pcie_gen, r.pcie_width),
            r.started_at, serde_json::to_string(&r.meta).unwrap(),
            v.tier, serde_json::to_string(&v.notes).unwrap(), sha,
            f.machine.as_ref().map(|m| m.to_string()), f.selection.as_ref().map(|m| m.to_string())
        ],
    )
    .map_err(|e| e.to_string())?;
    {
        let mut app = tx.appender("points").map_err(|e| e.to_string())?;
        for model::Row(dir, size, cfg, gbps, n, cv, base, sel) in &f.rows {
            app.append_row(params![r.id, dir, size, cfg, gbps, n, cv, base, sel]).map_err(|e| e.to_string())?;
        }
    }
    tx.commit().map_err(|e| e.to_string())?;
    Ok((r.id.clone(), v.tier, f.rows.len()))
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

// ---------------------------------------------------------------- read side

pub use crate::data::{Curve, RunRow};

pub fn load_runs(c: &Connection) -> duckdb::Result<Vec<RunRow>> {
    let mut st = c.prepare(
        "SELECT r.id, r.machine, coalesce(r.gpu,''), coalesce(r.cpu,''), r.tier, r.status, coalesce(r.note,''),
                r.pcie_gen, r.pcie_width, r.link_gbps, r.started_at, r.verify_tier, coalesce(r.verify_notes,'[]'),
                coalesce(r.source_sha256,''),
                (SELECT count(*) FROM points p WHERE p.run_id = r.id),
                (SELECT count(DISTINCT config) FROM points p WHERE p.run_id = r.id),
                r.selection IS NOT NULL
         FROM runs r ORDER BY r.machine, r.id",
    )?;
    let rows = st.query_map([], |r| {
        let gen: Option<u8> = r.get(7)?;
        let w: Option<u8> = r.get(8)?;
        let notes: String = r.get(12)?;
        Ok(RunRow {
            id: r.get(0)?,
            machine: r.get(1)?,
            gpu: r.get(2)?,
            cpu: r.get(3)?,
            tier: r.get(4)?,
            status: r.get(5)?,
            note: r.get(6)?,
            pcie: gen.map(|g| format!("Gen{g} x{}", w.unwrap_or(16))),
            link_gbps: r.get(9)?,
            started_at: r.get(10)?,
            verify: r.get(11)?,
            verify_notes: serde_json::from_str(&notes).unwrap_or_default(),
            sha256: r.get(13)?,
            n_points: r.get(14)?,
            n_configs: r.get(15)?,
            has_selection: r.get(16)?,
        })
    })?;
    rows.collect()
}

pub fn load_curves(c: &Connection) -> duckdb::Result<BTreeMap<String, BTreeMap<String, Curve>>> {
    let mut st = c.prepare(
        "WITH t AS (
           SELECT run_id, dir, size,
                  coalesce(max(gbps) FILTER (WHERE selected), max(gbps)) AS best,
                  coalesce(arg_max(config, gbps) FILTER (WHERE selected), arg_max(config, gbps)) AS cfg,
                  max(gbps) FILTER (WHERE baseline) AS base
           FROM points GROUP BY ALL)
         SELECT run_id, dir, size, best, cfg, base FROM t ORDER BY run_id, dir, size",
    )?;
    let mut out: BTreeMap<String, BTreeMap<String, Curve>> = BTreeMap::new();
    let mut rows = st.query([])?;
    while let Some(r) = rows.next()? {
        let cur = out.entry(r.get(0)?).or_default().entry(r.get(1)?).or_default();
        cur.size.push(r.get(2)?);
        cur.best.push(round3(r.get(3)?));
        cur.base.push(r.get::<_, Option<f64>>(5)?.map(round3));
        cur.cfg.push(r.get(4)?);
    }
    Ok(out)
}

pub fn run_meta(c: &Connection, id: &str) -> duckdb::Result<Value> {
    let m: String = c.query_row("SELECT coalesce(meta,'{}') FROM runs WHERE id=?", params![id], |r| r.get(0))?;
    let mut m: Value = serde_json::from_str(&m).unwrap_or(json!({}));
    if let Some(o) = m.as_object_mut() {
        o.retain(|k, _| scrub::public_meta_key(k));
    }
    Ok(m)
}

pub fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

/// Full run in the import format (round-trips through `cmx import`).
pub fn export_meta(c: &Connection, run: &RunRow) -> duckdb::Result<Value> {
    let meta = run_meta(c, &run.id)?;
    let (gen, width) = c.query_row("SELECT pcie_gen, pcie_width FROM runs WHERE id=?", params![run.id], |r| {
        Ok((r.get::<_, Option<u8>>(0)?, r.get::<_, Option<u8>>(1)?))
    })?;
    let (machine, selection): (Option<String>, Option<String>) =
        c.query_row("SELECT machine_json, selection FROM runs WHERE id=?", params![run.id], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })?;
    let parse = |s: Option<String>| s.and_then(|s| serde_json::from_str::<Value>(&s).ok()).unwrap_or(Value::Null);
    let (machine, selection) = (scrub::public_machine(&parse(machine)), scrub::public_selection(&parse(selection)));
    let v = json!({
        "format": model::FORMAT,
        "run": {"id": run.id, "machine": run.machine, "gpu": run.gpu, "cpu": run.cpu, "tier": run.tier,
                "status": run.status, "note": run.note, "pcie_gen": gen, "pcie_width": width,
                "started_at": run.started_at, "meta": meta},
        "cols": model::COLS,
        "machine": machine,
        "selection": selection,
    });
    Ok(v)
}

pub fn row_point(r: &duckdb::Row) -> duckdb::Result<crate::data::Point> {
    Ok(crate::data::Point(
        r.get(0)?,
        r.get(1)?,
        r.get(2)?,
        round3(r.get(3)?),
        r.get(4)?,
        r.get::<_, Option<f64>>(5)?.map(|v| (v * 1e4).round() / 1e4),
        r.get(6)?,
        r.get(7)?,
    ))
}

pub fn top_points(c: &Connection, id: &str, k: i64) -> duckdb::Result<Vec<crate::data::Point>> {
    let mut st = c.prepare(
        "SELECT dir, size, config, gbps, n, cv, baseline, selected FROM (
           SELECT *, row_number() OVER (PARTITION BY dir, size ORDER BY gbps DESC) AS rk FROM points WHERE run_id=? AND NOT baseline)
         WHERE rk <= ? ORDER BY dir, size, gbps DESC",
    )?;
    let rows = st.query_map(params![id, k], row_point)?;
    rows.collect()
}
