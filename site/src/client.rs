//! Read static JSON once, then filter/group/page the decoded rows in the browser.
use crate::data::*;
use futures_util::{
    future::{LocalBoxFuture, Shared},
    FutureExt,
};
use serde::de::DeserializeOwned;
use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap},
    rc::Rc,
};
use wasm_bindgen::{JsCast, JsValue};

type RowsFuture = Shared<LocalBoxFuture<'static, Result<Rc<Vec<Point>>, String>>>;
thread_local! { static ROWS: RefCell<HashMap<String, RowsFuture>> = RefCell::new(HashMap::new()); }

fn directory(id: &str) -> Result<String, String> {
    if id.is_empty() || id == "." || id == ".." || !id.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
    {
        return Err("invalid run ID".into());
    }
    Ok(format!("/data/{id}"))
}

async fn fetch<T: DeserializeOwned>(path: &str) -> Result<T, String> {
    let response = gloo_net::http::Request::get(path).send().await.map_err(|e| e.to_string())?;
    if !response.ok() {
        return Err(format!("{path}: HTTP {}", response.status()));
    }
    serde_json::from_str(&response.text().await.map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

pub async fn get_summary() -> Result<Summary, String> {
    fetch("/data/summary.json").await
}
pub async fn get_detail(id: String) -> Result<Detail, String> {
    fetch(&format!("{}/detail.json", directory(&id)?)).await
}

async fn rows(id: String) -> Result<Rc<Vec<Point>>, String> {
    let key = id.clone();
    let future = ROWS.with(|cache| {
        cache
            .borrow_mut()
            .entry(id.clone())
            .or_insert_with(|| {
                async move {
                    let dir = directory(&id)?;
                    let packed: PackedPoints = fetch(&format!("{dir}/points.json")).await?;
                    let mut out = Vec::new();
                    for file in &packed.files {
                        if !file.starts_with("points-") || !file.ends_with(".json") || file.contains('/') {
                            return Err("invalid points filename".into());
                        }
                        let chunk: Vec<PackedPoint> = fetch(&format!("{dir}/{file}")).await?;
                        for PackedPoint(d, s, c, g, n, cv, flags) in chunk {
                            out.push(Point(
                                packed.dirs.get(d).ok_or("invalid direction index")?.clone(),
                                *packed.sizes.get(s).ok_or("invalid size index")?,
                                packed.configs.get(c).ok_or("invalid config index")?.clone(),
                                g,
                                n,
                                cv,
                                flags & 1 != 0,
                                flags & 2 != 0,
                            ));
                        }
                    }
                    Ok(Rc::new(out))
                }
                .boxed_local()
                .shared()
            })
            .clone()
    });
    let result = future.await;
    if result.is_err() {
        ROWS.with(|cache| cache.borrow_mut().remove(&key));
    }
    result
}

fn displayed(p: &Point) -> Point {
    Point(
        p.0.clone(),
        p.1,
        p.2.clone(),
        (p.3 * 1000.0).round() / 1000.0,
        p.4,
        p.5.map(|v| (v * 1e4).round() / 1e4),
        p.6,
        p.7,
    )
}

pub async fn get_points(id: String, q: PointsQuery) -> Result<Page, String> {
    let rows = rows(id).await?;
    let mut filtered: Vec<&Point> = rows
        .iter()
        .filter(|p| {
            (q.dir.is_empty() || q.dir.len() >= 16 || q.dir == p.0)
                && (q.size <= 0 || q.size == p.1)
                && (q.q.is_empty() || q.q.len() >= 64 || p.2.to_lowercase().contains(&q.q.to_lowercase()))
        })
        .collect();
    filtered.sort_by(|a, b| {
        let order = match q.sort.as_str() {
            "size" => a.1.cmp(&b.1),
            "config" => a.2.cmp(&b.2),
            "dir" => a.0.cmp(&b.0),
            _ => a.3.total_cmp(&b.3),
        };
        (if q.desc { order.reverse() } else { order }).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2))
    });
    let total = filtered.len() as i64;
    let page = q.page.clamp(0, 1_000_000);
    let per = q.per.clamp(1, 500);
    Ok(Page {
        total,
        page,
        per,
        rows: filtered.into_iter().skip((page * per) as usize).take(per as usize).map(displayed).collect(),
    })
}

pub async fn get_breakdown(id: String, q: BreakdownQuery) -> Result<Breakdown, String> {
    let rows = rows(id).await?;
    let mut groups: BTreeMap<String, BTreeMap<i64, f64>> = BTreeMap::new();
    let mut base: BTreeMap<i64, f64> = BTreeMap::new();
    for row in rows.iter().filter(|p| p.0 == q.dir) {
        if row.6 {
            let value = base.entry(row.1).or_insert(row.3);
            *value = value.max(row.3);
            continue;
        }
        let p = parse_cfg(&row.2);
        if p.kind == "cudaMemcpy" {
            continue;
        }
        let key = match q.by.as_str() {
            "isa" => p.isa,
            "threads" => p.threads,
            "buffers" => p.buffers,
            "size" => p.size,
            _ => p.kind,
        };
        if key.is_empty() {
            continue;
        }
        let value = groups.entry(key).or_default().entry(row.1).or_insert(row.3);
        *value = value.max(row.3);
    }
    let rank = |k: &str| {
        let mut it = k.split(' ');
        let n = it.next().and_then(|v| v.parse::<f64>().ok()).unwrap_or(f64::MAX);
        n * match it.next() {
            Some("KiB") => 1024.0,
            Some("MiB") => 1048576.0,
            Some("GiB") => 1073741824.0,
            _ => 1.0,
        }
    };
    let mut groups: Vec<(String, Vec<(i64, f64)>)> =
        groups.into_iter().map(|(k, v)| (k, v.into_iter().collect())).collect();
    groups.sort_by(|a, b| rank(&a.0).total_cmp(&rank(&b.0)).then(a.0.cmp(&b.0)));
    Ok(Breakdown { groups, base: base.into_iter().collect() })
}

pub async fn download_run(id: String, csv: bool) -> Result<(), String> {
    let data = rows(id.clone()).await?;
    let content = if csv {
        let mut text = String::from("run_id,dir,size_bytes,config,gbps,n,cv,baseline,selected\n");
        for p in data.iter() {
            let config =
                if p.2.contains([',', '"', '\n']) { format!("\"{}\"", p.2.replace('"', "\"\"")) } else { p.2.clone() };
            text.push_str(&format!(
                "{id},{},{},{config},{},{},{},{},{}\n",
                p.0,
                p.1,
                p.3,
                p.4.map(|v| v.to_string()).unwrap_or_default(),
                p.5.map(|v| v.to_string()).unwrap_or_default(),
                p.6,
                p.7
            ));
        }
        text
    } else {
        let mut document: serde_json::Value = fetch(&format!("{}/download.json", directory(&id)?)).await?;
        document["rows"] =
            serde_json::to_value(data.iter().map(displayed).collect::<Vec<_>>()).map_err(|e| e.to_string())?;
        document.to_string()
    };
    let options = web_sys::BlobPropertyBag::new();
    options.set_type(if csv { "text/csv;charset=utf-8" } else { "application/json" });
    let blob =
        web_sys::Blob::new_with_str_sequence_and_options(&js_sys::Array::of1(&JsValue::from_str(&content)), &options)
            .map_err(|e| format!("{e:?}"))?;
    let url = web_sys::Url::create_object_url_with_blob(&blob).map_err(|e| format!("{e:?}"))?;
    let window = web_sys::window().ok_or("no window")?;
    let document = window.document().ok_or("no document")?;
    let anchor = document.create_element("a").map_err(|e| format!("{e:?}"))?;
    anchor.set_attribute("href", &url).map_err(|e| format!("{e:?}"))?;
    anchor
        .set_attribute("download", &format!("{id}.{}", if csv { "csv" } else { "json" }))
        .map_err(|e| format!("{e:?}"))?;
    anchor.dyn_into::<web_sys::HtmlElement>().map_err(|_| "download element")?.click();
    let callback = wasm_bindgen::closure::Closure::once_into_js(move || {
        let _ = web_sys::Url::revoke_object_url(&url);
    });
    window
        .set_timeout_with_callback_and_timeout_and_arguments_0(callback.unchecked_ref(), 1000)
        .map_err(|e| format!("{e:?}"))?;
    Ok(())
}
