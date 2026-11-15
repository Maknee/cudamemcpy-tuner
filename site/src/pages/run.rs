use crate::chart::{LineChart, Series};
use crate::data::*;
use crate::Route;
use dioxus::prelude::*;
use std::collections::{BTreeMap, BTreeSet};

use super::{color_for, dir_order, line_sample};

#[component]
pub fn RunPage(id: String) -> Element {
    let id2 = id.clone();
    let detail = use_resource(move || get_detail(id2.clone()));
    let guard = detail.read();
    match guard.as_ref() {
        Some(Ok(d)) => rsx! { RunView { d: d.clone() } },
        Some(Err(_)) => {
            rsx! { div { class: "empty", "No run named “{id}”. " Link { to: Route::Home {}, "Back to all machines" } } }
        }
        None => rsx! { div { class: "empty", "Loading…" } },
    }
}

/// Config split into the tuner's grid dimensions, one cell each.
fn cfg_cells(label: &str) -> Element {
    let c = parse_cfg(label);
    if c.threads.is_empty() {
        return rsx! { td { class: "cfg", colspan: "5", title: "{label}", "{label}" } };
    }
    rsx! {
        td { class: "cfg", title: loop_help(&c.kind), "{c.kind}" if !c.extra.is_empty() { span { class: "muted", " {c.extra}" } } }
        td { class: "cfg", "{c.isa}" }
        td { class: "r num", "{c.threads}" }
        td { class: "r num", "{c.buffers}" }
        td { class: "r num", "{c.size}" }
    }
}

fn cfg_heads() -> Element {
    rsx! {
        th { title: "CPU copy loop into the pinned staging buffers", "Loop" }
        th { title: "instruction set used by the loop", "ISA" }
        th { class: "r", title: "CPU copy threads", "Threads" }
        th { class: "r", title: "staging buffers per thread", "Buffers" }
        th { class: "r", title: "size of each staging buffer", "Buffer size" }
    }
}

#[component]
fn RunView(d: Detail) -> Element {
    let r = d.run.clone();
    let mut download_error = use_signal(|| None::<String>);
    let hover = use_signal(|| None::<i64>);
    let pinned = use_signal(|| None::<i64>);
    let dirs: Vec<String> = {
        let mut v: Vec<String> = d.curves.keys().cloned().collect();
        v.sort_by_key(|x| dir_order(x));
        v
    };
    let mut series = vec![];
    let mut sizes = BTreeSet::new();
    for (i, dd) in dirs.iter().enumerate() {
        let c = &d.curves[dd];
        sizes.extend(c.size.iter().copied());
        series.push(Series {
            key: dd.clone(),
            color: color_for(i).into(),
            dashed: false,
            pts: c.size.iter().copied().zip(c.best.iter().copied()).collect(),
        });
        let b: Vec<(i64, f64)> = c.size.iter().zip(c.base.iter()).filter_map(|(s, b)| b.map(|b| (*s, b))).collect();
        if !b.is_empty() {
            series.push(Series { key: format!("{dd}-base"), color: color_for(i).into(), dashed: true, pts: b });
        }
    }
    let sizes: Vec<i64> = sizes.into_iter().collect();
    // fastest single config per (dir,size); `top` is sorted best-first
    let mut fastest: BTreeMap<(String, i64), f64> = BTreeMap::new();
    for p in &d.top {
        fastest.entry((p.0.clone(), p.1)).or_insert(p.3);
    }
    let sub = [r.gpu.trim_start_matches("NVIDIA ").to_string(), r.cpu.clone(), r.pcie.clone().unwrap_or_default()]
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" · ");

    rsx! {
        nav { class: "crumbs", Link { to: Route::Home {}, "← All machines" } }
        section { class: "hero",
            h1 { "{r.machine}" }
            p { class: "lede", "{sub}" }
            if !r.note.is_empty() { p { class: "note", "{r.note}" } }
            div { class: "links",
                button { class: "link", onclick: { let id = r.id.clone(); move |_| { let id = id.clone(); async move { download_error.set(download_run(id, false).await.err()); } } }, "Download JSON" }
                button { class: "link", onclick: { let id = r.id.clone(); move |_| { let id = id.clone(); async move { download_error.set(download_run(id, true).await.err()); } } }, "CSV" }
                if let Some(error) = download_error() { span { class: "warn-text", "Download failed: {error}" } }
            }
        }
        section { class: "panel",
            div { class: "legend",
                for (i, dd) in dirs.iter().enumerate() {
                    span { key: "{dd}", class: "key", span { class: "sw", style: "--c: {color_for(i)}" } "{dir_label(dd)}" }
                }
                span { class: "muted small", {line_sample(true, "cudaMemcpy")} }
            }
            LineChart { series, sizes, y_unit: "GB/s".to_string(), hover, pinned, ref_line: r.link_gbps, ref_label: "PCIe limit".to_string() }
        }
        Compare { id: r.id.clone(), dirs: dirs.clone() }
        for dd in dirs.iter() {
            section { class: "panel", key: "{dd}",
                h2 { "{dir_label(dd)}" }
                if r.has_selection {
                    p { class: "muted small",
                        "Tuner pick: the config the tuner chose for this size range. Best measured: the fastest single config at exactly this size."
                    }
                }
                div { class: "scroll",
                    table { class: "grid-table",
                        thead { tr {
                            th { class: "r", "Size" }
                            th { class: "r", if r.has_selection { "Tuner pick" } else { "Best measured" } }
                            if r.has_selection { th { class: "r", "Best measured" } }
                            th { class: "r", "cudaMemcpy" }
                            th { class: "r", "Speedup" }
                            {cfg_heads()}
                        } }
                        tbody {
                            for k in 0..d.curves[dd].size.len() {
                                {
                                    let c = &d.curves[dd];
                                    let (sz, best, base, cfg) = (c.size[k], c.best[k], c.base[k], c.cfg[k].clone());
                                    let hot = hover().or(pinned()) == Some(sz);
                                    // `top` excludes plain cudaMemcpy rows, which the tuner may itself have picked
                                    let top = fastest.get(&(dd.clone(), sz)).map(|t| t.max(best));
                                    rsx! {
                                        tr { key: "{sz}", class: if hot { "hot" } else { "" },
                                            td { class: "r num", {fmt_size(sz)} }
                                            td { class: "r num strong", {fmt_gbps(best)} }
                                            if r.has_selection {
                                                td { class: if top.is_some_and(|t| t > best * 1.05) { "r num warn-text" } else { "r num muted" }, {top.map(fmt_gbps).unwrap_or("—".into())} }
                                            }
                                            td { class: "r num", {base.map(fmt_gbps).unwrap_or("—".into())} }
                                            td { class: "r num", {base.filter(|b| *b > 0.0).map(|b| format!("{:.2}×", best / b)).unwrap_or("—".into())} }
                                            {cfg_cells(&cfg)}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        AllPoints { id: r.id.clone(), dirs: dirs.clone(), sizes: d.curves.values().flat_map(|c| c.size.iter().copied()).collect::<BTreeSet<i64>>().into_iter().collect::<Vec<i64>>() }
        details { class: "panel",
            summary { "Machine and run details" }
            div { class: "kv",
                span { "Checks" }
                b {
                    if r.verify_notes.is_empty() { "All automated checks passed" } else { {r.verify_notes.join("; ")} }
                }
                span { "Status" } b { "{r.status}" }
                span { "Measurements" } b { "{r.n_points} across {r.n_configs} configs" }
                if let Some(l) = r.link_gbps { span { "PCIe limit" } b { "{l:.1} GB/s" } }
                for (k, v) in d.meta.iter().filter(|(k, _)| k.as_str() != "link") {
                    span { key: "k{k}", "{k}" } b { "{v}" }
                }
                span { "Run id" } b { class: "mono", "{r.id}" }
                span { "sha256" } b { class: "mono", "{r.sha256}" }
            }
        }
    }
}

const DIMS: [(&str, &str); 5] =
    [("size", "Buffer size"), ("threads", "Threads"), ("buffers", "Buffers"), ("loop", "Loop"), ("isa", "ISA")];

/// Best GB/s reached by each value of one config dimension, across every measured config.
#[component]
fn Compare(id: String, dirs: Vec<String>) -> Element {
    let first = dirs.first().cloned().unwrap_or_default();
    let mut q = use_signal(|| BreakdownQuery { dir: first, by: "size".into() });
    let cur = q();
    rsx! {
        section { class: "panel",
            div { class: "panel-head",
                h2 { "Compare configs" }
            }
            div { class: "filters inline",
                div { class: "seg", role: "group",
                    for dd in dirs.iter().cloned() {
                        button { key: "{dd}", class: if dd == cur.dir { "on" } else { "" }, onclick: move |_| { let mut v = q(); v.dir = dd.clone(); q.set(v) }, {dir_label(&dd)} }
                    }
                }
                label { class: "field",
                    span { class: "lbl", "Lines by" }
                    select {
                        onchange: move |e| { let mut v = q(); v.by = e.value(); q.set(v) },
                        for (k, l) in DIMS {
                            option { key: "{k}", value: "{k}", selected: cur.by == k, "{l}" }
                        }
                    }
                }
            }
            SuspenseBoundary {
                fallback: |_| rsx! { div { class: "empty", "Loading…" } },
                CompareChart { id: id.clone(), q }
            }
            div { class: "muted small chart-note",
                span { "Each line is the best GB/s any config with that setting reached." }
                {line_sample(true, "cudaMemcpy")}
            }
        }
    }
}

#[component]
fn CompareChart(id: String, q: Signal<BreakdownQuery>) -> Element {
    let id2 = id.clone();
    let res = use_resource(move || get_breakdown(id2.clone(), q()));
    let hover = use_signal(|| None::<i64>);
    let pinned = use_signal(|| None::<i64>);
    let guard = res.read();
    let b = match guard.as_ref() {
        Some(Ok(b)) => b,
        None => return rsx! { div { class: "empty", "Loading…" } },
        Some(Err(error)) => return rsx! { div { class: "empty", "Could not load: {error}" } },
    };
    let mut sizes = BTreeSet::new();
    let mut series = vec![];
    for (i, (k, pts)) in b.groups.iter().enumerate() {
        sizes.extend(pts.iter().map(|p| p.0));
        series.push(Series { key: k.clone(), color: color_for(i).into(), dashed: false, pts: pts.clone() });
    }
    if !b.base.is_empty() {
        sizes.extend(b.base.iter().map(|p| p.0));
        series.push(Series { key: "base".into(), color: "var(--quiet)".into(), dashed: true, pts: b.base.clone() });
    }
    let at = hover().or(pinned());
    rsx! {
        div { class: "legend",
            for (i, (k, pts)) in b.groups.iter().enumerate() {
                span { key: "{k}", class: "key",
                    span { class: "sw", style: "--c: {color_for(i)}" }
                    "{k}"
                    if let Some(v) = at.and_then(|a| pts.iter().find(|p| p.0 == a)) {
                        span { class: "num muted", " {fmt_gbps(v.1)}" }
                    }
                }
            }
        }
        LineChart { series, sizes: sizes.into_iter().collect::<Vec<_>>(), y_unit: "GB/s".to_string(), hover, pinned }
    }
}

const PER: i64 = 50;

fn upd(mut q: Signal<PointsQuery>, f: impl FnOnce(&mut PointsQuery)) {
    let mut v = q();
    f(&mut v);
    q.set(v);
}

#[component]
fn AllPoints(id: String, dirs: Vec<String>, sizes: Vec<i64>) -> Element {
    let q = use_signal(|| PointsQuery { sort: "gbps".into(), desc: true, per: PER, ..Default::default() });
    rsx! {
        section { class: "panel",
            h2 { "All measurements" }
            div { class: "filters inline",
                select {
                    onchange: move |e| { let v = e.value(); upd(q, |x| { x.dir = v.clone(); x.page = 0; }) },
                    option { value: "", "Both directions" }
                    for dd in dirs.iter() { option { key: "{dd}", value: "{dd}", {dir_label(dd)} } }
                }
                select {
                    onchange: move |e| { let v = e.value().parse().unwrap_or(0); upd(q, |x| { x.size = v; x.page = 0; }) },
                    option { value: "0", "All sizes" }
                    for s in sizes.iter() { option { key: "{s}", value: "{s}", {fmt_size(*s)} } }
                }
                input {
                    r#type: "search",
                    placeholder: "Filter configs, e.g. avx512",
                    onchange: move |e| { let v = e.value(); upd(q, |x| { x.q = v.clone(); x.page = 0; }) },
                }
                select {
                    onchange: move |e| { let v = e.value(); upd(q, |x| { x.sort = v.clone(); x.page = 0; }) },
                    option { value: "gbps", "Sort by GB/s" }
                    option { value: "size", "Sort by size" }
                    option { value: "config", "Sort by config" }
                }
                button { class: "btn", onclick: move |_| upd(q, |x| { x.desc = !x.desc; x.page = 0; }), if q().desc { "↓ High first" } else { "↑ Low first" } }
            }
            SuspenseBoundary {
                fallback: |_| rsx! { div { class: "empty", "Loading…" } },
                PointsTable { id: id.clone(), q }
            }
        }
    }
}

#[component]
fn PointsTable(id: String, q: Signal<PointsQuery>) -> Element {
    let id2 = id.clone();
    let page = use_resource(move || get_points(id2.clone(), q()));
    let guard = page.read();
    let p = match guard.as_ref() {
        Some(Ok(p)) => p,
        None => return rsx! { div { class: "empty", "Loading…" } },
        Some(Err(error)) => return rsx! { div { class: "empty", "Could not load measurements: {error}" } },
    };
    if p.total == 0 {
        return rsx! { div { class: "empty", "No measurements match." } };
    }
    let pages = ((p.total + p.per - 1) / p.per).max(1);
    let cur = p.page;
    rsx! {
        div { class: "scroll",
            table { class: "grid-table dense",
                thead { tr {
                    th { "Direction" } th { class: "r", "Size" } {cfg_heads()} th { class: "r", "GB/s" } th { class: "r", title: "samples", "n" } th { class: "r", title: "spread between samples", "CV" }
                } }
                tbody {
                    for (i, pt) in p.rows.iter().enumerate() {
                        tr { key: "{i}", class: if pt.7 { "picked" } else if pt.6 { "base" } else { "" },
                            td { {dir_label(&pt.0)} }
                            td { class: "r num", {fmt_size(pt.1)} }
                            {cfg_cells(&pt.2)}
                            td { class: "r num strong", {fmt_gbps(pt.3)} if pt.7 { span { class: "tag", title: "the tuner's pick for this size", "pick" } } }
                            td { class: "r num muted", {pt.4.map(|v| v.to_string()).unwrap_or_default()} }
                            td { class: "r num muted", {pt.5.map(|v| format!("{:.1}%", v * 100.0)).unwrap_or_default()} }
                        }
                    }
                }
            }
        }
        div { class: "pager",
            button { class: "btn", disabled: cur == 0, onclick: move |_| { let mut v = q(); v.page = (cur - 1).max(0); q.set(v) }, "‹ Prev" }
            span { class: "num muted", "{cur + 1} / {pages} · {p.total} rows" }
            button { class: "btn", disabled: cur + 1 >= pages, onclick: move |_| { let mut v = q(); v.page = cur + 1; q.set(v) }, "Next ›" }
        }
    }
}
