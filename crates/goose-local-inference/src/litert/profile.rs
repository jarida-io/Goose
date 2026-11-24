//! LiteRT-LM's [`ModelProfile`]: what the file says through the C API, what it stores (see
//! [`super::file`]), and what this backend does with it. The backend sends text only, so the
//! model card lists no other input even where the file carries encoders.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::SystemTime;

use goose_provider_types::canonical::{CanonicalModel, Limit, Modalities, Modality};
use goose_provider_types::errors::ProviderError;
use goose_provider_types::model_profile::{
    Accelerator, Caches, ChosenBy, Dtype, Endpoint, KvCache, KvLayout, ModelProfile, Precision,
    PrefixCache, PrefixReuse, Rewind, Runtime, Setting, SideCalls, SnapshotCache, SnapshotContents,
    ToolChoiceSupport,
};

use super::ffi;
use super::file::{self, KvInput, LitertlmFile};
use crate::local_model_registry::SamplingConfig;
use crate::ResolvedModelPaths;

pub(super) fn profile(
    model_id: &str,
    resolved: &ResolvedModelPaths,
) -> Result<ModelProfile, ProviderError> {
    let settings = &resolved.settings;
    let litert = settings.litert.clone().unwrap_or_default();
    let execution = super::execution_backend(litert.backend.as_deref());
    let (library_dir, giap) = ffi::library()?;

    let loaded = ffi::LoadedFile::open(&resolved.model_path)?;
    let file_context = loaded.max_context_tokens();
    let max_num_tokens = super::max_num_tokens(
        settings.context_size,
        super::env_context_limit(),
        file_context,
    );
    let speculative_decoding =
        litert.speculative_decoding.unwrap_or(true) && loaded.supports_speculative_decoding();
    let reasoning = loaded.supports_thinking();
    let tool_call = loaded.supports_function_calling();
    drop(loaded);

    let file = stored(&resolved.model_path);
    let mut quantization = file.quantization.clone();
    quantization.activations = activations(&execution, giap, file.prefer_activation.as_deref());
    let key = file.kv.iter().find(|input| input.is_key());
    let value = file.kv.iter().find(|input| !input.is_key());
    quantization.kv_key = key.map(fixed_by_file);
    quantization.kv_value = value.map(fixed_by_file);

    let (layout, bytes_per_token, fixed_bytes) = kv_layout(&file.kv);
    let caches = Caches {
        kv: KvCache {
            layout,
            capacity_tokens: Some(max_num_tokens),
            bytes_per_token,
            fixed_bytes,
            // Rematch rewinds by moving the step counter, which a ring buffer does not follow.
            rewind: match (layout, giap) {
                (KvLayout::Static, true) => Rewind::Free,
                _ => Rewind::None,
            },
        },
        prefix: PrefixCache {
            reuse: if giap {
                PrefixReuse::LongestPrefix
            } else {
                PrefixReuse::AppendOnly
            },
            side_calls: if giap {
                SideCalls::SetAsideToDisk
            } else {
                SideCalls::CopiedInMemory
            },
            through_media: false,
        },
        snapshot: giap.then_some(SnapshotCache {
            holds: SnapshotContents::WholeConversation,
            personal: true,
            kept: Some(super::SNAPSHOTS_KEPT),
            owner_only: false,
        }),
        compiled_programs: true,
    };

    let runtime = Runtime {
        backend: super::LITERT_BACKEND_ID.to_string(),
        accelerator: accelerator(&execution, library_dir),
        speculative_decoding,
    };

    let model = CanonicalModel {
        id: model_id.to_string(),
        name: model_id.to_string(),
        family: None,
        attachment: Some(false),
        reasoning: Some(reasoning),
        thinking_mode: None,
        tool_call,
        temperature: Some(true),
        knowledge: None,
        release_date: None,
        last_updated: None,
        modalities: Modalities {
            input: vec![Modality::Text],
            output: vec![Modality::Text],
        },
        open_weights: Some(true),
        cost: Default::default(),
        limit: Limit {
            context: file_context.map_or(max_num_tokens, |n| n as usize),
            output: settings.max_output_tokens,
        },
    };

    let mut supported_parameters = vec!["tools", "max_tokens", "seed"];
    if matches!(settings.sampling, SamplingConfig::Temperature { .. }) {
        supported_parameters.extend(["temperature", "top_k", "top_p"]);
    }
    let endpoint = Endpoint {
        quantization: quantization.openrouter_bucket().to_string(),
        context_length: Some(max_num_tokens),
        max_completion_tokens: settings.max_output_tokens,
        supported_parameters: supported_parameters
            .into_iter()
            .map(str::to_string)
            .collect(),
        // LiteRT-LM ends a turn early only by cancelling it, which discards the conversation.
        supports_tool_choice: ToolChoiceSupport {
            none: true,
            auto: true,
            required: false,
            function: false,
        },
        supports_implicit_caching: true,
    };

    Ok(ModelProfile {
        model,
        endpoint,
        quantization,
        caches,
        runtime,
    })
}

/// The file's stored facts, read once per file version: the decoder's tables are spread
/// through the whole file.
fn stored(path: &Path) -> Arc<LitertlmFile> {
    type Key = (PathBuf, u64, Option<SystemTime>);
    static READ: OnceLock<Mutex<HashMap<Key, Arc<LitertlmFile>>>> = OnceLock::new();
    let metadata = std::fs::metadata(path).ok();
    let key = (
        path.to_path_buf(),
        metadata.as_ref().map_or(0, |m| m.len()),
        metadata.and_then(|m| m.modified().ok()),
    );
    let cache = READ.get_or_init(Default::default);
    if let Some(file) = cache.lock().ok().and_then(|c| c.get(&key).cloned()) {
        return file;
    }
    let file = Arc::new(file::read(path).unwrap_or_else(|error| {
        tracing::warn!(
            backend = super::LITERT_BACKEND_ID,
            file = %path.display(),
            %error,
            "Could not read what the LiteRT-LM file stores"
        );
        LitertlmFile::default()
    }));
    if let Ok(mut cache) = cache.lock() {
        cache.insert(key, Arc::clone(&file));
    }
    file
}

fn fixed_by_file(input: &KvInput) -> Setting<Precision> {
    Setting {
        value: Precision::plain(input.dtype, input.bits_per_value),
        chosen_by: ChosenBy::ModelFile,
        options: Vec::new(),
    }
}

/// On the GPU, giap-main computes in fp32 whatever the file prefers (fp16 corrupted tool calls
/// there); otherwise the file's preference holds.
fn activations(execution: &str, giap: bool, preferred: Option<&str>) -> Option<Setting<Dtype>> {
    if execution == "gpu" && giap {
        return Some(Setting {
            value: Dtype::Fp32,
            chosen_by: ChosenBy::Engine,
            options: Vec::new(),
        });
    }
    let preferred = preferred?.to_ascii_lowercase();
    let value = if preferred.starts_with("fp32") {
        Dtype::Fp32
    } else if preferred.starts_with("fp16") {
        Dtype::Fp16
    } else if preferred.starts_with("bf16") {
        Dtype::Bf16
    } else {
        return None;
    };
    Some(Setting {
        value,
        chosen_by: ChosenBy::ModelFile,
        options: Vec::new(),
    })
}

/// Inputs sized to the context placeholder keep every position; shorter ones keep a window.
fn kv_layout(kv: &[KvInput]) -> (KvLayout, Option<u64>, Option<u64>) {
    let Some(full) = kv.iter().map(KvInput::positions).max() else {
        return (KvLayout::Static, None, None);
    };
    let bytes_per_token = kv
        .iter()
        .filter(|input| input.positions() == full)
        .map(KvInput::bytes_per_position)
        .sum();
    let windowed: Vec<&KvInput> = kv.iter().filter(|input| input.positions() < full).collect();
    let Some(window) = windowed.iter().map(|input| input.positions()).min() else {
        return (KvLayout::Static, Some(bytes_per_token), None);
    };
    let fixed_bytes = windowed
        .iter()
        .map(|input| input.positions() * input.bytes_per_position())
        .sum();
    let keys = |sliding: bool| {
        kv.iter()
            .filter(|input| input.is_key() && (input.positions() < full) == sliding)
            .count() as u64
    };
    (
        KvLayout::Hybrid {
            window,
            sliding_layers: keys(true),
            full_layers: keys(false),
        },
        Some(bytes_per_token),
        Some(fixed_bytes),
    )
}

/// LiteRT loads its GPU accelerator from the library's directory, trying WebGPU before Metal on
/// macOS; on Linux the GPU path is WebGPU over Vulkan.
fn accelerator(execution: &str, library_dir: &Path) -> Accelerator {
    if execution != "gpu" {
        return Accelerator::Cpu;
    }
    let has = |stem: &str| {
        ["dylib", "so", "dll"]
            .iter()
            .any(|ext| library_dir.join(format!("{stem}.{ext}")).is_file())
    };
    if has("libLiteRtWebGpuAccelerator") {
        Accelerator::WebGpu
    } else if has("libLiteRtMetalAccelerator") {
        Accelerator::Metal
    } else {
        Accelerator::Cpu
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(name: &str, shape: &[i64]) -> KvInput {
        KvInput {
            name: name.to_string(),
            dtype: Dtype::Int8,
            bits_per_value: 8.0,
            shape: shape.to_vec(),
        }
    }

    #[test]
    fn caches_sized_to_the_context_are_static() {
        let kv = [
            input("kv_cache_k_0", &[1, 1, 32003, 256]),
            input("kv_cache_v_0", &[1, 1, 256, 32003]),
            input("kv_cache_k_1", &[1, 1, 32003, 512]),
            input("kv_cache_v_1", &[1, 1, 512, 32003]),
        ];
        let (layout, per_token, fixed) = kv_layout(&kv);
        assert_eq!(layout, KvLayout::Static);
        assert_eq!(per_token, Some(256 + 256 + 512 + 512));
        assert_eq!(fixed, None);
    }

    #[test]
    fn shorter_caches_are_a_sliding_window() {
        let kv = [
            input("kv_cache_k_0", &[1, 1, 512, 256]),
            input("kv_cache_v_0", &[1, 1, 256, 512]),
            input("kv_cache_k_1", &[1, 1, 32003, 512]),
            input("kv_cache_v_1", &[1, 1, 512, 32003]),
        ];
        let (layout, per_token, fixed) = kv_layout(&kv);
        assert_eq!(
            layout,
            KvLayout::Hybrid {
                window: 512,
                sliding_layers: 1,
                full_layers: 1
            }
        );
        assert_eq!(per_token, Some(1024));
        assert_eq!(fixed, Some(512 * 256 * 2));
    }

    #[test]
    fn the_gpu_computes_in_fp32_with_giap_main_and_follows_the_file_otherwise() {
        let gpu = activations("gpu", true, Some("fp16")).unwrap();
        assert_eq!((gpu.value, gpu.chosen_by), (Dtype::Fp32, ChosenBy::Engine));
        let upstream = activations("gpu", false, Some("fp16")).unwrap();
        assert_eq!(
            (upstream.value, upstream.chosen_by),
            (Dtype::Fp16, ChosenBy::ModelFile)
        );
        assert_eq!(
            activations("cpu", true, Some("fp32_fp16")).unwrap().value,
            Dtype::Fp32
        );
        assert!(activations("cpu", true, None).is_none());
    }

    #[test]
    fn the_accelerator_is_the_one_the_library_directory_carries() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(accelerator("cpu", dir.path()), Accelerator::Cpu);
        assert_eq!(accelerator("gpu", dir.path()), Accelerator::Cpu);
        std::fs::write(dir.path().join("libLiteRtMetalAccelerator.dylib"), b"").unwrap();
        assert_eq!(accelerator("gpu", dir.path()), Accelerator::Metal);
        std::fs::write(dir.path().join("libLiteRtWebGpuAccelerator.dylib"), b"").unwrap();
        assert_eq!(accelerator("gpu", dir.path()), Accelerator::WebGpu);
    }
}
