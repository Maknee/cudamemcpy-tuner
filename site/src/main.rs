#![allow(non_snake_case)]
//! Static benchmark explorer; the export feature builds the offline data tool.

#[cfg(not(feature = "export"))]
mod chart;
#[cfg(feature = "web")]
mod client;
#[cfg_attr(feature = "export", allow(dead_code))]
mod data;
#[cfg(not(feature = "export"))]
mod pages;
#[cfg(not(feature = "export"))]
mod url;

#[cfg(feature = "export")]
mod backend;

#[cfg(not(feature = "export"))]
use dioxus::prelude::*;
#[cfg(not(feature = "export"))]
use pages::{Home, NotFound, RunPage};

#[cfg(not(feature = "export"))]
const CSS: &str = include_str!("style.css");

// Both website links follow this URL if the repository moves.
#[cfg(not(feature = "export"))]
const GITHUB_REPOSITORY: &str = "https://github.com/Maknee/cudamemcpy-tuner";

#[cfg(not(feature = "export"))]
#[derive(Routable, Clone, PartialEq, Debug)]
#[rustfmt::skip]
pub enum Route {
    #[layout(Shell)]
        #[route("/")]
        Home {},
        #[route("/run/:id/")]
        RunPage { id: String },
    #[end_layout]
    #[route("/:..rest")]
    NotFound { rest: Vec<String> },
}

#[cfg(feature = "export")]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    std::process::exit(backend::cli(&args));
}

#[cfg(not(feature = "export"))]
fn main() {
    dioxus::launch(App);
}

#[cfg(not(feature = "export"))]
fn App() -> Element {
    rsx! {
        document::Title { "cudamemcpy-tuner benchmarks" }
        document::Meta { name: "description", content: "Host↔GPU copy bandwidth: cudamemcpy-tuner's tuned copy vs plain cudaMemcpy, across GPUs and buffer sizes." }
        document::Meta { name: "viewport", content: "width=device-width, initial-scale=1" }
        style { dangerous_inner_html: CSS }
        Router::<Route> {}
    }
}

#[cfg(not(feature = "export"))]
#[component]
fn Shell() -> Element {
    rsx! {
        div { class: "root",
            header { class: "top",
                Link { class: "brand", to: Route::Home {},
                    "cudamemcpy"
                    span { class: "tld", ".com" }
                }
                nav { class: "project-links", aria_label: "Project",
                    a { href: GITHUB_REPOSITORY, "GitHub" }
                    a { href: "{GITHUB_REPOSITORY}/tree/HEAD/tunes", "Submit a tune (PR)" }
                }
            }
            main { Outlet::<Route> {} }
            footer {
                span { "GB/s = 10⁹ bytes/s. Sizes in KiB/MiB/GiB." }
                Link { to: Route::Home {}, "Browse benchmarks" }
            }
        }
    }
}
