use crate::chart::{LineChart, Series};
use crate::data::*;
use crate::{url, Route};
use dioxus::prelude::*;
use std::collections::{BTreeMap, BTreeSet};

use super::{color_for, dir_order, line_sample};

#[component]
pub fn Home() -> Element {
    let summary = use_resource(get_summary);
    let guard = summary.read();
    match guard.as_ref() {
        Some(Ok(s)) => rsx! { Explorer { s: s.clone() } },
        Some(Err(e)) => rsx! { div { class: "empty", "Failed to load: {e}" } },
        None => rsx! { div { class: "empty", "Loading…" } },
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Metric {
    Gbps,
    Speedup,
    Link,
}
impl Metric {
    fn key(self) -> &'static str {
        match self {
            Metric::Gbps => "gbps",
            Metric::Speedup => "speedup",
            Metric::Link => "link",
        }
    }
    fn label(self) -> &'static str {
        match self {
            Metric::Gbps => "GB/s",
            Metric::Speedup => "Speedup",
            Metric::Link => "% of PCIe",
        }
    }
    fn from(s: &str) -> Self {
        match s {
            "speedup" => Metric::Speedup,
            "link" => Metric::Link,
            _ => Metric::Gbps,
        }
    }
}

#[component]
fn Explorer(s: Summary) -> Element {
    let dirs: Vec<String> = {
        let mut d: BTreeSet<String> = BTreeSet::new();
        for c in s.curves.values() {
            d.extend(c.keys().cloned());
        }
        let mut v: Vec<String> = d.into_iter().collect();
        v.sort_by_key(|x| dir_order(x));
        v
    };
    let latest = latest_per_machine(&s.runs);

    let first_dir = dirs.first().cloned().unwrap_or_default();
    let mut dir = use_signal(|| first_dir.clone());
    let mut metric = use_signal(|| Metric::Gbps);
    let mut gpu = use_signal(BTreeSet::<String>::new);
    let mut cpu = use_signal(BTreeSet::<String>::new);
    let mut find = use_signal(String::new);
    let mut hidden = use_signal(BTreeSet::<String>::new);
    let mut show_base = use_signal(|| true);
    let hover = use_signal(|| None::<i64>);
    let mut pinned = use_signal(|| None::<i64>);

    // URL -> state once after hydration; state -> URL on every change (defaults stay out of the URL)
    let mut synced = use_signal(|| false);
    use_effect(move || {
        if let Some(d) = url::get("dir") {
            dir.set(d);
        }
        if let Some(m) = url::get("metric") {
            metric.set(Metric::from(&m));
        }
        if let Some(g) = url::get("gpu") {
            gpu.set(g.split(',').map(str::to_string).collect());
        }
        if let Some(c) = url::get("cpu") {
            cpu.set(c.split(',').map(str::to_string).collect());
        }
        if url::get("base").as_deref() == Some("0") {
            show_base.set(false);
        }
        if let Some(p) = url::get("at").and_then(|v| v.parse().ok()) {
            pinned.set(Some(p));
        }
        synced.set(true);
    });
    let fd = first_dir.clone();
    use_effect(move || {
        let kv = [
            ("dir", if dir() == fd { String::new() } else { dir() }),
            ("metric", if metric() == Metric::Gbps { String::new() } else { metric().key().into() }),
            ("gpu", gpu().into_iter().collect::<Vec<_>>().join(",")),
            ("cpu", cpu().into_iter().collect::<Vec<_>>().join(",")),
            ("base", if show_base() { String::new() } else { "0".into() }),
            ("at", pinned().map(|v| v.to_string()).unwrap_or_default()),
        ];
        if *synced.peek() {
            url::set(&kv);
        }
    });

    let d = dir();
    let m = metric();
    let g = gpu();
    let cp = cpu();
    let count = |it: &mut dyn Iterator<Item = String>| {
        let mut m: BTreeMap<String, usize> = BTreeMap::new();
        for k in it {
            *m.entry(k).or_default() += 1;
        }
        m.into_iter().collect::<Vec<_>>()
    };
    let gpu_ok = |r: &RunRow| g.is_empty() || g.contains(&gpu_type(&r.gpu));
    let cpu_ok = |r: &RunRow| cp.is_empty() || cp.contains(&cpu_short(&r.cpu));
    // every value stays listed; the count says how many runs it adds given the other filter
    let all_of = |f: &dyn Fn(&RunRow) -> String| count(&mut s.runs.iter().map(|r| f(r)));
    let with_zero = |all: Vec<(String, usize)>, hits: Vec<(String, usize)>| -> Vec<(String, usize)> {
        all.into_iter()
            .map(|(k, _)| {
                let n = hits.iter().find(|h| h.0 == k).map(|h| h.1).unwrap_or(0);
                (k, n)
            })
            .collect()
    };
    let gpus = with_zero(
        all_of(&|r| gpu_type(&r.gpu)),
        count(&mut s.runs.iter().filter(|r| cpu_ok(r)).map(|r| gpu_type(&r.gpu))),
    );
    let cpus = with_zero(
        all_of(&|r| cpu_short(&r.cpu)),
        count(&mut s.runs.iter().filter(|r| gpu_ok(r)).map(|r| cpu_short(&r.cpu))),
    );
    // no filter = newest run per machine; any filter = every run that matches
    let shown: Vec<(usize, &RunRow)> = s
        .runs
        .iter()
        .enumerate()
        .filter(|(_, r)| if g.is_empty() && cp.is_empty() { latest.contains(&r.id) } else { gpu_ok(r) && cpu_ok(r) })
        .collect();
    let needle = find().to_lowercase();
    let catalog: Vec<&RunRow> = s
        .runs
        .iter()
        .filter(|r| {
            needle.is_empty() || [&r.machine, &r.gpu, &r.cpu, &r.id].iter().any(|f| f.to_lowercase().contains(&needle))
        })
        .collect();
    let visible: Vec<(usize, &RunRow, &Curve)> = shown
        .iter()
        .filter(|(_, r)| !hidden().contains(&r.id))
        .filter_map(|(i, r)| s.curves.get(&r.id).and_then(|c| c.get(&d)).map(|c| (*i, *r, c)))
        .collect();

    let value = |r: &RunRow, best: f64, base: Option<f64>| -> Option<f64> {
        match m {
            Metric::Gbps => Some(best),
            Metric::Speedup => base.filter(|b| *b > 0.0).map(|b| best / b),
            Metric::Link => r.link_gbps.map(|l| best / l * 100.0),
        }
    };
    let mut series = vec![];
    let mut all_sizes = BTreeSet::new();
    for (i, r, c) in &visible {
        let mut pts = vec![];
        let mut bpts = vec![];
        for k in 0..c.size.len() {
            all_sizes.insert(c.size[k]);
            if let Some(v) = value(r, c.best[k], c.base[k]) {
                pts.push((c.size[k], v));
            }
            if let Some(b) = c.base[k] {
                match m {
                    Metric::Gbps => bpts.push((c.size[k], b)),
                    Metric::Link => {
                        if let Some(l) = r.link_gbps {
                            bpts.push((c.size[k], b / l * 100.0))
                        }
                    }
                    Metric::Speedup => {}
                }
            }
        }
        series.push(Series { key: r.id.clone(), color: color_for(*i).into(), dashed: false, pts });
        if show_base() && !bpts.is_empty() {
            series.push(Series { key: format!("{}-base", r.id), color: color_for(*i).into(), dashed: true, pts: bpts });
        }
    }
    let sizes: Vec<i64> = all_sizes.into_iter().collect();
    let at = hover().or(pinned()).or_else(|| sizes.last().copied());
    let (ref_line, ref_label) = match m {
        Metric::Speedup => (Some(1.0), "cudaMemcpy"),
        Metric::Link => (Some(100.0), "PCIe limit"),
        Metric::Gbps => (None, ""),
    };
    let any_best_of = visible.iter().any(|(_, r, _)| !r.has_selection);

    // table at the active size, sorted by the active metric
    let mut rows: Vec<(usize, &RunRow, Option<(f64, Option<f64>, String)>)> = visible
        .iter()
        .map(|(i, r, c)| {
            let v =
                at.and_then(|a| c.size.iter().position(|x| *x == a)).map(|k| (c.best[k], c.base[k], c.cfg[k].clone()));
            (*i, *r, v)
        })
        .collect();
    let key = |r: &RunRow, v: &Option<(f64, Option<f64>, String)>| {
        v.as_ref().and_then(|(b, base, _)| value(r, *b, *base)).unwrap_or(-1.0)
    };
    rows.sort_by(|a, b| key(b.1, &b.2).total_cmp(&key(a.1, &a.2)));

    rsx! {
        section { class: "hero",
            h1 { "cudamemcpy-tuner benchmarks" }
        }
        section { class: "filters",
            MultiSelect { label: "GPU", options: gpus, sel: gpu, hidden }
            MultiSelect { label: "CPU", options: cpus, sel: cpu, hidden }
            div { class: "seg", role: "group",
                for dd in dirs.iter().cloned() {
                    button { key: "{dd}", class: if dd == d { "on" } else { "" }, onclick: move |_| dir.set(dd.clone()), {dir_label(&dd)} }
                }
            }
            div { class: "seg", role: "group",
                for mm in [Metric::Gbps, Metric::Speedup, Metric::Link] {
                    button { key: "{mm.key()}", class: if mm == m { "on" } else { "" }, onclick: move |_| metric.set(mm), {mm.label()} }
                }
            }
            label { class: if m == Metric::Speedup { "check dim" } else { "check" },
                input { r#type: "checkbox", checked: show_base(), disabled: m == Metric::Speedup, onchange: move |e| show_base.set(e.checked()) }
                "cudaMemcpy line"
            }
        }
        section { class: "panel",
            div { class: "legend",
                for (i, r) in shown.iter().copied() {
                    button {
                        key: "{r.id}",
                        class: if hidden().contains(&r.id) { "key off" } else { "key" },
                        title: "click to hide / show",
                        onclick: {
                            let id = r.id.clone();
                            move |_| {
                                let mut h = hidden();
                                if !h.remove(&id) {
                                    h.insert(id.clone());
                                }
                                hidden.set(h);
                            }
                        },
                        span { class: "sw", style: "--c: {color_for(i)}" }
                        "{r.machine}"
                    }
                }
            }
            LineChart { series, sizes: sizes.clone(), y_unit: m.label().to_string(), hover, pinned, ref_line, ref_label: ref_label.to_string(), tight: m == Metric::Speedup }
            div { class: "muted small chart-note",
                {line_sample(false, "Tuner's pick")}
                {line_sample(true, "cudaMemcpy")}
                if any_best_of { span { "Runs without a tuner pick show their fastest measured config." } }
            }
        }
        section { class: "panel",
            div { class: "panel-head",
                h2 { span { class: "num", {at.map(fmt_size).unwrap_or_default()} } " · {dir_label(&d)}" }
                if pinned().is_some() {
                    button { class: "link", onclick: move |_| pinned.set(None), "unpin" }
                }
            }
            div { class: "scroll",
                table { class: "grid-table",
                    thead {
                        tr {
                            th { "Machine" }
                            th { class: "r", "Tuned GB/s" }
                            th { class: "r", "cudaMemcpy" }
                            th { class: "r", "Speedup" }
                            th { class: "r", "% of PCIe" }
                            th { "Config" }
                        }
                    }
                    tbody {
                        for (i, r, v) in rows.iter() {
                            tr { key: "{r.id}",
                                td {
                                    span { class: "sw", style: "--c: {color_for(*i)}" }
                                    Link { to: Route::RunPage { id: r.id.clone() }, "{r.machine}" }
                                }
                                if let Some((best, base, cfg)) = v {
                                    td { class: "r num strong", {fmt_gbps(*best)} }
                                    td { class: "r num", {base.map(fmt_gbps).unwrap_or("—".into())} }
                                    td { class: "r num", {base.filter(|b| *b > 0.0).map(|b| format!("{:.2}×", best / b)).unwrap_or("—".into())} }
                                    td { class: "r num", {r.link_gbps.map(|l| format!("{:.0}%", best / l * 100.0)).unwrap_or("—".into())} }
                                    td { class: "cfg", title: "{cfg}", {cfg_short(cfg)} }
                                } else {
                                    td { class: "muted", colspan: "5", "not measured at this size" }
                                }
                            }
                        }
                    }
                }
            }
        }
        section { class: "panel", id: "runs",
            h2 { "Runs" }
            input {
                class: "search",
                r#type: "search",
                "aria-label": "Search runs",
                placeholder: "Search runs by GPU or CPU, e.g. 5090 or Xeon",
                value: "{find}",
                oninput: move |e| find.set(e.value()),
            }
            div { class: "scroll",
                table { class: "grid-table",
                    thead { tr { th { "Run" } th { "Date" } th { "PCIe" } } }
                    tbody {
                        if catalog.is_empty() {
                            tr { td { class: "muted", colspan: "3", "No runs match." } }
                        }
                        for r in catalog.iter() {
                            tr { key: "{r.id}",
                                td { Link { to: Route::RunPage { id: r.id.clone() }, "{r.machine}" } }
                                td { class: "muted num", {run_date(&r.id)} }
                                td { class: "muted num", {r.pcie.clone().unwrap_or_default()} }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Checkbox dropdown; empty selection means "all".
#[component]
fn MultiSelect(
    label: &'static str,
    options: Vec<(String, usize)>,
    sel: Signal<BTreeSet<String>>,
    hidden: Signal<BTreeSet<String>>,
) -> Element {
    let cur = sel();
    let summary = match cur.len() {
        0 => "All".to_string(),
        1 => cur.iter().next().unwrap().clone(),
        n => format!("{n} selected"),
    };
    let mut open = use_signal(|| false);
    let mut close = move || {
        open.set(false);
        url::blur();
    };
    rsx! {
        div {
            class: if open() { "ms open" } else { "ms" },
            onkeydown: move |e| {
                if e.key() == Key::Escape {
                    close();
                }
            },
            button { class: "ms-btn", "aria-expanded": open(), onclick: move |_| if open() { close() } else { open.set(true) },
                span { class: "lbl", "{label}" }
                " {summary}"
            }
            if open() {
                // invisible layer behind the menu: any click outside closes it
                div { class: "ms-backdrop", onclick: move |_| close() }
            }
            div { class: "menu", hidden: !open(),
                for (t, n) in options.into_iter() {
                    label { key: "{t}", class: if n == 0 && !cur.contains(&t) { "opt none" } else { "opt" },
                        input {
                            r#type: "checkbox",
                            checked: cur.contains(&t),
                            onchange: {
                                let t = t.clone();
                                move |_| {
                                    let mut s = sel();
                                    if !s.remove(&t) {
                                        s.insert(t.clone());
                                    }
                                    sel.set(s);
                                    hidden.set(BTreeSet::new());
                                }
                            },
                        }
                        span { "{t}" }
                        span { class: "muted num", "{n}" }
                    }
                }
                if !cur.is_empty() {
                    button { class: "link", onclick: move |_| { sel.set(BTreeSet::new()); hidden.set(BTreeSet::new()); }, "Clear" }
                }
            }
        }
    }
}
