//! Where the multimodal encoder runs, in mtmd's own words. mtmd reports its backend, flash
//! attention and any operation its GPU backend lacks only at info and warning level, which a
//! production log keeps quiet, so the lines are caught while the encoder loads and summarised.

use std::ffi::{c_char, c_void, CStr};
use std::sync::Mutex;

static CAUGHT: Mutex<Vec<String>> = Mutex::new(Vec::new());

unsafe extern "C" fn catch(
    _level: llama_cpp_sys_2::ggml_log_level,
    text: *const c_char,
    _user_data: *mut c_void,
) {
    if text.is_null() {
        return;
    }
    // SAFETY: mtmd passes a NUL-terminated string that lives for the call.
    let line = unsafe { CStr::from_ptr(text) }.to_string_lossy();
    CAUGHT
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(line.into_owned());
}

/// Run `load` with mtmd's log caught, then hand mtmd's log back to tracing.
pub(super) fn caught_while<T>(load: impl FnOnce() -> T) -> (T, Vec<String>) {
    CAUGHT.lock().unwrap_or_else(|e| e.into_inner()).clear();
    // SAFETY: `catch` only reads the text it is given; no user data is passed.
    unsafe { llama_cpp_sys_2::mtmd_helper_log_set(Some(catch), std::ptr::null_mut()) };
    let out = load();
    // Reinstalls the same tracing bridge goose set up at backend init.
    llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default());
    let lines = std::mem::take(&mut *CAUGHT.lock().unwrap_or_else(|e| e.into_inner()));
    (out, lines)
}

/// What one encoder load said about where it runs: one entry per tower it set up.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct EncoderBackend {
    pub backends: Vec<String>,
    pub flash_attention: Vec<String>,
    /// Operations the GPU backend cannot run, which fall back to the CPU.
    pub cpu_ops: Vec<String>,
}

impl EncoderBackend {
    pub fn from_lines(text: &str) -> EncoderBackend {
        let mut out = EncoderBackend::default();
        let mut listing_ops = false;
        for line in text.lines() {
            let body = line.split_once(": ").map_or(line, |(_, rest)| rest).trim();
            if let Some(rest) = body.strip_prefix("CLIP using ") {
                out.backends
                    .push(rest.trim_end_matches(" backend").trim().to_string());
            } else if let Some(state) = body.strip_prefix("flash attention is ") {
                listing_ops = false;
                let state = state.trim().to_string();
                if !out.flash_attention.contains(&state) {
                    out.flash_attention.push(state);
                }
            } else if body.starts_with("list of unsupported ops") {
                listing_ops = true;
            } else if listing_ops {
                if let Some((op, _)) = body.split_once(": type = ") {
                    let op = op.trim().to_string();
                    if !out.cpu_ops.contains(&op) {
                        out.cpu_ops.push(op);
                    }
                }
            }
        }
        out
    }

    /// Every tower on a GPU backend, with nothing left for the CPU.
    pub fn fully_accelerated(&self) -> bool {
        !self.backends.is_empty()
            && self.backends.iter().all(|b| !b.eq_ignore_ascii_case("CPU"))
            && self.cpu_ops.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// mtmd's lines as clip.cpp prints them for an audio tower on CUDA with two ops it lacks.
    const CUDA_WITH_GAPS: &str = "\
clip_model_loader: model name:   Gemma 4 E4B
clip_init: CLIP using CUDA0 backend
warmup: flash attention is enabled
warmup: *****************************************************************
warmup: WARNING: the CLIP graph uses unsupported operators by the backend
warmup:          the performance will be suboptimal
warmup:          list of unsupported ops (backend=CUDA0):
warmup:          CONV_1D: type = f32, ne = [3000 128 1 1]
warmup:              PAD: type = f32, ne = [3010 128 1 1]
warmup:          CONV_1D: type = f32, ne = [1500 128 1 1]
warmup: flash attention is enabled
warmup: please report this on github as an issue
";

    #[test]
    fn reads_the_backend_flash_attention_and_the_ops_left_to_the_cpu() {
        let seen = EncoderBackend::from_lines(CUDA_WITH_GAPS);
        assert_eq!(seen.backends, vec!["CUDA0"]);
        assert_eq!(seen.flash_attention, vec!["enabled"]);
        assert_eq!(seen.cpu_ops, vec!["CONV_1D", "PAD"]);
        assert!(!seen.fully_accelerated());
    }

    #[test]
    fn a_tower_wholly_on_the_gpu_is_fully_accelerated_and_the_cpu_never_is() {
        let gpu = EncoderBackend::from_lines(
            "clip_init: CLIP using MTL0 backend\nwarmup: flash attention is enabled\n",
        );
        assert!(gpu.fully_accelerated());
        let cpu = EncoderBackend::from_lines(
            "clip_init: CLIP using CPU backend\nwarmup: flash attention is disabled\n",
        );
        assert_eq!(cpu.backends, vec!["CPU"]);
        assert!(!cpu.fully_accelerated());
        assert!(!EncoderBackend::from_lines("").fully_accelerated());
    }
}
