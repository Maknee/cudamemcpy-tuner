//! Browser query-string state, viewport width and focus.

#[cfg(feature = "web")]
mod imp {
    fn params() -> Option<web_sys::UrlSearchParams> {
        let s = web_sys::window()?.location().search().ok()?;
        web_sys::UrlSearchParams::new_with_str(&s).ok()
    }
    pub fn get(k: &str) -> Option<String> {
        params()?.get(k).filter(|v| !v.is_empty())
    }
    /// Replace the given keys in the query string without adding history entries.
    pub fn set(kv: &[(&str, String)]) {
        let Some(w) = web_sys::window() else { return };
        let Some(p) = params() else { return };
        for (k, v) in kv {
            if v.is_empty() {
                p.delete(k);
            } else {
                p.set(k, v);
            }
        }
        let path = w.location().pathname().unwrap_or_default();
        let qs: String = p.to_string().into();
        let url = if qs.is_empty() { path } else { format!("{path}?{qs}") };
        if let Ok(h) = w.history() {
            let _ = h.replace_state_with_url(&wasm_bindgen::JsValue::NULL, "", Some(&url));
        }
    }
    pub fn narrow() -> bool {
        web_sys::window().and_then(|w| w.inner_width().ok()).and_then(|v| v.as_f64()).is_some_and(|w| w < 640.0)
    }
    /// Drop keyboard focus so no focus ring lingers on a closed control.
    pub fn blur() {
        use wasm_bindgen::JsCast;
        if let Some(el) = web_sys::window().and_then(|w| w.document()).and_then(|d| d.active_element()) {
            if let Ok(h) = el.dyn_into::<web_sys::HtmlElement>() {
                let _ = h.blur();
            }
        }
    }
}

#[cfg(not(feature = "web"))]
mod imp {
    pub fn get(_: &str) -> Option<String> {
        None
    }
    pub fn set(_: &[(&str, String)]) {}
    pub fn narrow() -> bool {
        false
    }
    pub fn blur() {}
}

pub use imp::*;
