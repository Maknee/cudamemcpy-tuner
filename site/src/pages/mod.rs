//! The two routes and their small shared presentation helpers.

mod explorer;
mod run;
pub use explorer::Home;
pub use run::RunPage;

use crate::{data::PALETTE, Route};
use dioxus::prelude::*;

fn color_for(i: usize) -> &'static str {
    PALETTE[i % PALETTE.len()]
}

fn dir_order(x: &str) -> u8 {
    match x {
        "h2d" => 0,
        "d2h" => 1,
        _ => 2,
    }
}

/// A short line in the chart's stroke style, followed by its meaning.
fn line_sample(dashed: bool, text: &str) -> Element {
    rsx! {
        span { class: "sample",
            svg { width: "26", height: "8", view_box: "0 0 26 8",
                line { x1: "1", x2: "25", y1: "4", y2: "4", stroke: "currentColor", stroke_width: if dashed { "1.5" } else { "2.2" }, stroke_dasharray: if dashed { "4 3" } else { "none" }, stroke_linecap: "round" }
            }
            "{text}"
        }
    }
}

#[component]
pub fn NotFound(rest: Vec<String>) -> Element {
    rsx! {
        div { class: "root",
            main { div { class: "empty", "Nothing at /{rest.join(\"/\")}. " Link { to: Route::Home {}, "Go home" } } }
        }
    }
}
