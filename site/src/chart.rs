//! Dependency-free SVG line chart: log2 x (bytes), linear y. Hover = per-size bands.

use crate::data::fmt_size;
use dioxus::prelude::*;

#[derive(Clone, PartialEq)]
pub struct Series {
    pub key: String,
    pub color: String,
    pub dashed: bool,
    pub pts: Vec<(i64, f64)>,
}

const L: f64 = 44.0;
const R: f64 = 20.0;
const T: f64 = 26.0;
const B: f64 = 34.0;

/// Nice axis [lo, hi] with ~5 steps.
fn nice(lo: f64, hi: f64) -> (f64, f64, f64) {
    let span = (hi - lo).max(1e-9);
    let raw = span / 5.0;
    let mag = 10f64.powf(raw.log10().floor());
    let step = [1.0, 2.0, 2.5, 5.0, 10.0].iter().map(|m| m * mag).find(|s| *s >= raw).unwrap_or(10.0 * mag);
    ((lo / step).floor() * step, (hi / step).ceil() * step, step)
}

fn tick_label(v: f64, step: f64) -> String {
    if step < 0.1 {
        format!("{v:.2}")
    } else if step < 1.0 {
        format!("{v:.1}")
    } else {
        format!("{v:.0}")
    }
}

#[component]
pub fn LineChart(
    series: Vec<Series>,
    sizes: Vec<i64>,
    y_unit: String,
    hover: Signal<Option<i64>>,
    pinned: Signal<Option<i64>>,
    #[props(default)] ref_line: Option<f64>,
    #[props(default)] ref_label: String,
    /// start the y axis near the data instead of at 0 (ratios)
    #[props(default)]
    tight: bool,
) -> Element {
    // SSR renders the desktop shape; phones switch to a narrower box after hydration so text stays legible
    let mut narrow = use_signal(|| false);
    use_effect(move || narrow.set(crate::url::narrow()));
    let (W, H) = if narrow() { (400.0, 300.0) } else { (960.0, 340.0) };
    if sizes.is_empty() {
        return rsx! { div { class: "empty", "Nothing selected." } };
    }
    let lx0 = (*sizes.first().unwrap() as f64).log2();
    let lx1 = (*sizes.last().unwrap() as f64).log2().max(lx0 + 1.0);
    let vals = || series.iter().flat_map(|s| s.pts.iter().map(|p| p.1));
    let dmax = vals().fold(f64::MIN, f64::max).max(ref_line.unwrap_or(f64::MIN));
    let dmin = vals().fold(f64::MAX, f64::min).min(ref_line.unwrap_or(f64::MAX));
    let (dmax, dmin) = if dmax == f64::MIN { (1.0, 0.0) } else { (dmax, dmin) };
    let (ymin, ymax, ystep) = if tight {
        let pad = (dmax - dmin).max(0.05) * 0.06;
        nice((dmin - pad).max(0.0), dmax + pad)
    } else {
        nice(0.0, dmax * 1.04)
    };
    let x = move |s: i64| L + ((s as f64).log2() - lx0) / (lx1 - lx0) * (W - L - R);
    let y = move |v: f64| T + (1.0 - (v - ymin) / (ymax - ymin)) * (H - T - B);

    let yticks: Vec<f64> = (0..=(((ymax - ymin) / ystep).round() as i64)).map(|i| ymin + i as f64 * ystep).collect();
    let span = (lx1 - lx0).max(1.0);
    let per_tick = span / ((W - L - R) / 60.0);
    let step = [1, 2, 4, 8].into_iter().find(|s| *s as f64 >= per_tick).unwrap_or(8);
    let xticks: Vec<i64> = sizes
        .iter()
        .copied()
        .filter(|s| (s.trailing_zeros() as i64) % step == 0 && (*s as u64).is_power_of_two())
        .collect();

    // hover bands: midpoints between neighbouring sizes
    let bands: Vec<(i64, f64, f64)> = sizes
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let a = if i == 0 { L } else { (x(sizes[i - 1]) + x(s)) / 2.0 };
            let b = if i + 1 == sizes.len() { W - R } else { (x(s) + x(sizes[i + 1])) / 2.0 };
            (s, a, b - a)
        })
        .collect();
    let active = hover().or(pinned());

    rsx! {
        svg {
            class: "chart",
            view_box: "0 0 {W} {H}",
            role: "img",
            onmouseleave: move |_| hover.set(None),
            g { class: "grid",
                for t in yticks.iter().copied() {
                    line { key: "y{t}", x1: L, x2: W - R, y1: y(t), y2: y(t) }
                    text { x: L - 8.0, y: y(t) + 4.0, text_anchor: "end", {tick_label(t, ystep)} }
                }
                for s in xticks.iter().copied() {
                    text { key: "x{s}", x: x(s), y: H - B + 18.0, text_anchor: "middle", {fmt_size(s)} }
                }
                text { x: 4.0, y: 12.0, class: "unit", "{y_unit}" }
            }
            if let Some(r) = ref_line.filter(|r| *r <= ymax) {
                line { class: "ref", x1: L, x2: W - R, y1: y(r), y2: y(r) }
                text { class: "reflabel", x: W - R - 4.0, y: y(r) - 5.0, text_anchor: "end", "{ref_label}" }
            }
            if let Some(a) = active {
                line { class: "cross", x1: x(a), x2: x(a), y1: T, y2: H - B }
            }
            for s in series.iter() {
                path {
                    key: "{s.key}",
                    class: if s.dashed { "line dashed" } else { "line" },
                    stroke: "{s.color}",
                    d: {
                        // break the line across sizes this series never measured
                        let idx = |sz: i64| sizes.binary_search(&sz).unwrap_or(0);
                        let mut d = String::with_capacity(s.pts.len() * 16);
                        for (i, (sz, v)) in s.pts.iter().enumerate() {
                            let joined = i > 0 && idx(*sz) == idx(s.pts[i - 1].0) + 1;
                            d.push_str(&format!("{}{:.1},{:.1}", if joined { "L" } else { "M" }, x(*sz), y(*v)));
                        }
                        d
                    },
                }
                if let Some(a) = active {
                    if let Some((_, v)) = s.pts.iter().find(|p| p.0 == a) {
                        circle { key: "{s.key}-dot", cx: x(a), cy: y(*v), r: if s.dashed { 2.5 } else { 3.5 }, fill: "{s.color}" }
                    }
                }
            }
            g { class: "bands",
                for (s, bx, bw) in bands.iter().copied() {
                    rect {
                        key: "b{s}",
                        x: bx,
                        y: T,
                        width: bw,
                        height: H - T - B,
                        onmouseenter: move |_| hover.set(Some(s)),
                        onclick: move |_| pinned.set(if pinned() == Some(s) { None } else { Some(s) }),
                    }
                }
            }
        }
    }
}
