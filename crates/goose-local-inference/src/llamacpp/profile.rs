//! llama.cpp's [`ModelProfile`], from the GGUF headers of the model, its encoder and its drafter
//! and from the model's settings, without loading any weights.

use std::path::Path;

use goose_provider_types::canonical::{CanonicalModel, Limit, Modalities, Modality};
use goose_provider_types::errors::ProviderError;
use goose_provider_types::model_profile::{
    Accelerator, Caches, ChosenBy, Component, Dtype, Endpoint, KvCache, KvLayout, ModelProfile,
    Precision, PrefixCache, PrefixReuse, Quantization, Rewind, Runtime, Setting, SideCalls,
    SnapshotCache, SnapshotContents, ToolChoiceSupport,
};
use llama_cpp_2::{LlamaBackendDevice, LlamaBackendDeviceType};

use super::memory::{bytes_per_32, GgufHeader, KvMeta};
use crate::local_model_registry::{KvCachePolicy, ModelSettings, SamplingConfig, ToolCallingMode};
use crate::ResolvedModelPaths;

/// The KV types llama.cpp's `--cache-type-k` and `--cache-type-v` take. A quantised V cache
/// needs flash attention.
const KV_TYPES: [&str; 9] = [
    "f32", "f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl",
];

pub(super) fn profile(
    model_id: &str,
    resolved: &ResolvedModelPaths,
    devices: &[LlamaBackendDevice],
) -> Result<ModelProfile, ProviderError> {
    let settings = &resolved.settings;
    let header = GgufHeader::open(&resolved.model_path).map_err(ProviderError::ExecutionError)?;
    let architecture = header.string("general.architecture");
    let trained_context = architecture
        .as_deref()
        .and_then(|arch| header.uint(&format!("{arch}.context_length")))
        .and_then(|n| usize::try_from(n).ok());

    let mut quantization = Quantization::default();
    add_tensors(&mut quantization, &header, component_of);
    drop(header);
    let hears = resolved
        .mmproj_path
        .as_deref()
        .map(|mmproj| add_file(&mut quantization, mmproj, encoder_component_of))
        .is_some_and(|encoders| encoders.contains(&Component::AudioEncoder));
    let sees = quantization.weights.contains_key(&Component::VisionEncoder);
    if let Some(drafter) = &resolved.draft_model_path {
        add_file(&mut quantization, drafter, |_| Component::Drafter);
    }

    let kv_choice = |named: &Option<String>| Setting {
        value: kv_precision(named.as_deref().unwrap_or("f16")),
        chosen_by: match (named, settings.kv_cache_policy) {
            (_, KvCachePolicy::Q8WhenTight) => ChosenBy::Engine,
            (Some(_), KvCachePolicy::Fixed) => ChosenBy::Settings,
            (None, KvCachePolicy::Fixed) => ChosenBy::Engine,
        },
        options: KV_TYPES.iter().map(|name| kv_precision(name)).collect(),
    };
    quantization.kv_key = Some(kv_choice(&settings.type_k));
    quantization.kv_value = Some(kv_choice(&settings.type_v));

    let geometry = KvMeta::read(&resolved.model_path)
        .ok()
        .and_then(|meta| meta.geometry());
    let cost = geometry.map(|g| {
        g.cost(
            settings.type_k.as_deref(),
            settings.type_v.as_deref(),
            settings.n_ubatch,
        )
    });
    let layout = match geometry {
        Some(g) if g.sliding_layers > 0 => KvLayout::Hybrid {
            window: g.n_swa,
            sliding_layers: g.sliding_layers,
            full_layers: g.full_layers,
        },
        _ => KvLayout::Static,
    };
    let context_length = settings.context_size.map(|n| n as usize);
    let caches = Caches {
        kv: KvCache {
            layout,
            capacity_tokens: context_length,
            bytes_per_token: cost.map(|c| c.bytes_per_token),
            fixed_bytes: cost.map(|c| c.sliding_bytes).filter(|&bytes| bytes > 0),
            rewind: match layout {
                KvLayout::Hybrid { .. } => Rewind::Checkpointed,
                KvLayout::Static => Rewind::Free,
            },
        },
        prefix: PrefixCache {
            reuse: PrefixReuse::LongestPrefix,
            side_calls: SideCalls::OwnContext,
            through_media: sees || hears,
        },
        // `prompt_snapshot` keeps only the beginning every prompt for the model shared; a turn
        // set aside for a helper is held apart from it, owner-only, and deleted on resume.
        snapshot: Some(SnapshotCache {
            holds: SnapshotContents::StablePrefix,
            personal: false,
            kept: None,
            owner_only: false,
        }),
        compiled_programs: false,
    };

    let accelerator = accelerator(settings.n_gpu_layers, devices);
    let runtime = Runtime {
        backend: super::LLAMACPP_BACKEND_ID.to_string(),
        accelerator,
        speculative_decoding: cfg!(feature = "cuda") && resolved.draft_model_path.is_some(),
    };

    let mut input = vec![Modality::Text];
    if sees {
        input.push(Modality::Image);
    }
    if hears {
        input.push(Modality::Audio);
    }
    let model = CanonicalModel {
        id: model_id.to_string(),
        name: model_id.to_string(),
        family: architecture,
        attachment: Some(sees || hears),
        reasoning: Some(settings.enable_thinking),
        thinking_mode: None,
        tool_call: !matches!(settings.tool_calling, ToolCallingMode::ForceEmulated),
        temperature: Some(true),
        knowledge: None,
        release_date: None,
        last_updated: None,
        modalities: Modalities {
            input,
            output: vec![Modality::Text],
        },
        open_weights: Some(true),
        cost: Default::default(),
        limit: Limit {
            context: trained_context.unwrap_or_default(),
            output: settings.max_output_tokens,
        },
    };

    let endpoint = Endpoint {
        quantization: quantization.openrouter_bucket().to_string(),
        context_length,
        max_completion_tokens: settings.max_output_tokens,
        supported_parameters: supported_parameters(settings),
        // `required` through `request_context::tool_calls_only`, which stops at the first prose.
        supports_tool_choice: ToolChoiceSupport {
            none: true,
            auto: true,
            required: true,
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

fn supported_parameters(settings: &ModelSettings) -> Vec<String> {
    let mut names = vec!["tools", "max_tokens", "seed"];
    if matches!(settings.sampling, SamplingConfig::Temperature { .. }) {
        names.extend(["temperature", "top_k", "top_p", "min_p"]);
    }
    names.extend([
        "repetition_penalty",
        "frequency_penalty",
        "presence_penalty",
    ]);
    names.into_iter().map(str::to_string).collect()
}

/// The accelerator llama.cpp offloads to: none when the settings keep every layer on the CPU,
/// else the first GPU ggml found. `None` layers is llama.cpp's default, which offloads.
fn accelerator(n_gpu_layers: Option<u32>, devices: &[LlamaBackendDevice]) -> Accelerator {
    if n_gpu_layers == Some(0) {
        return Accelerator::Cpu;
    }
    devices
        .iter()
        .filter(|d| {
            matches!(
                d.device_type,
                LlamaBackendDeviceType::Gpu | LlamaBackendDeviceType::IntegratedGpu
            )
        })
        .find_map(|d| accelerator_named(&d.backend))
        .unwrap_or(Accelerator::Cpu)
}

fn accelerator_named(backend: &str) -> Option<Accelerator> {
    let backend = backend.to_ascii_lowercase();
    if backend.contains("cuda") {
        Some(Accelerator::Cuda)
    } else if backend.contains("metal") || backend.starts_with("mtl") {
        Some(Accelerator::Metal)
    } else if backend.contains("vulkan") {
        Some(Accelerator::Vulkan)
    } else {
        None
    }
}

fn component_of(tensor: &str) -> Component {
    if tensor.starts_with("per_layer_token_embd") {
        Component::PerLayerEmbeddings
    } else if tensor.starts_with("token_embd") {
        Component::TokenEmbeddings
    } else {
        Component::Decoder
    }
}

/// The towers `encoder_towers` splits an mmproj into; `mm.` is the picture projector.
fn encoder_component_of(tensor: &str) -> Component {
    if tensor.starts_with("a.") || tensor.starts_with("mm.a.") {
        Component::AudioEncoder
    } else {
        Component::VisionEncoder
    }
}

/// Adds the GGUF at `path`'s weights; the components found, empty when it cannot be read.
fn add_file(
    quantization: &mut Quantization,
    path: &Path,
    component: fn(&str) -> Component,
) -> Vec<Component> {
    match GgufHeader::open(path) {
        Ok(header) => add_tensors(quantization, &header, component),
        Err(_) => Vec::new(),
    }
}

fn add_tensors(
    quantization: &mut Quantization,
    header: &GgufHeader,
    component: fn(&str) -> Component,
) -> Vec<Component> {
    let mut found = Vec::new();
    for tensor in header.tensors() {
        let Some(precision) = ggml_precision(&tensor.type_name, tensor.bits_per_value) else {
            continue;
        };
        let part = component(&tensor.name);
        quantization.add_weights(part, precision, tensor.bytes);
        if !found.contains(&part) {
            found.push(part);
        }
    }
    found
}

fn kv_precision(name: &str) -> Precision {
    let bits = bytes_per_32(Some(name)) as f32 * 8.0 / 32.0;
    let bits = if name.eq_ignore_ascii_case("bf16") {
        16.0
    } else {
        bits
    };
    ggml_precision(name, bits).unwrap_or_else(|| Precision::plain(Dtype::Fp16, 16.0))
}

/// A ggml type as a [`Precision`]; `None` for the integer index types, which hold no weights.
fn ggml_precision(type_name: &str, bits_per_value: f32) -> Option<Precision> {
    let lower = type_name.to_ascii_lowercase();
    let dtype = match lower.as_str() {
        "f32" => return Some(Precision::plain(Dtype::Fp32, bits_per_value)),
        "f16" => return Some(Precision::plain(Dtype::Fp16, bits_per_value)),
        "bf16" => return Some(Precision::plain(Dtype::Bf16, bits_per_value)),
        "i8" => return Some(Precision::plain(Dtype::Int8, bits_per_value)),
        "mxfp4" => Dtype::Fp4,
        s if s.starts_with("q8") => Dtype::Int8,
        s if s.starts_with("q6") => Dtype::Int6,
        s if s.starts_with("q5") => Dtype::Int5,
        s if s.starts_with("q4") || s.starts_with("iq4") => Dtype::Int4,
        s if s.starts_with("q3") || s.starts_with("iq3") => Dtype::Int3,
        s if s.starts_with("q2") || s.starts_with("iq2") || s.starts_with("tq2") => Dtype::Int2,
        s if s.starts_with("iq1") || s.starts_with("tq1") => Dtype::Int1,
        _ => return None,
    };
    Some(Precision::with_scheme(
        dtype,
        &type_name.to_ascii_uppercase(),
        bits_per_value,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ggml_block_formats_fall_into_their_width_and_keep_their_name() {
        let q4 = ggml_precision("q4_K", 4.5).unwrap();
        assert_eq!(q4.dtype, Dtype::Int4);
        assert_eq!(q4.scheme.as_deref(), Some("Q4_K"));
        assert_eq!(ggml_precision("iq4_xs", 4.25).unwrap().dtype, Dtype::Int4);
        assert_eq!(ggml_precision("q6_K", 6.5625).unwrap().dtype, Dtype::Int6);
        assert_eq!(ggml_precision("iq2_xxs", 2.06).unwrap().dtype, Dtype::Int2);
        assert_eq!(ggml_precision("mxfp4", 4.25).unwrap().dtype, Dtype::Fp4);
        assert_eq!(ggml_precision("f16", 16.0).unwrap().scheme, None);
        assert!(ggml_precision("i32", 32.0).is_none());
    }

    #[test]
    fn kv_types_carry_their_block_cost() {
        let q8 = kv_precision("q8_0");
        assert_eq!((q8.dtype, q8.bits_per_value), (Dtype::Int8, 8.5));
        assert_eq!(kv_precision("f16").bits_per_value, 16.0);
        assert_eq!(kv_precision("bf16").dtype, Dtype::Bf16);
        assert_eq!(kv_precision("q4_0").bits_per_value, 4.5);
    }

    #[test]
    fn tensors_are_sorted_into_components_by_name() {
        assert_eq!(
            component_of("token_embd.weight"),
            Component::TokenEmbeddings
        );
        assert_eq!(
            component_of("per_layer_token_embd.weight"),
            Component::PerLayerEmbeddings
        );
        assert_eq!(component_of("blk.0.attn_q.weight"), Component::Decoder);
        assert_eq!(component_of("output_norm.weight"), Component::Decoder);
        assert_eq!(encoder_component_of("a.blk.0.w"), Component::AudioEncoder);
        assert_eq!(
            encoder_component_of("mm.a.input_projection.weight"),
            Component::AudioEncoder
        );
        assert_eq!(
            encoder_component_of("mm.input_projection.weight"),
            Component::VisionEncoder
        );
        assert_eq!(encoder_component_of("v.blk.0.w"), Component::VisionEncoder);
    }

    /// Profiles a real file when `GGUF_TEST_FILE` names one (and `GGUF_TEST_MMPROJ` its encoder).
    #[test]
    fn profiles_a_real_file_when_given_one() {
        let Some(path) = std::env::var_os("GGUF_TEST_FILE") else {
            return;
        };
        let resolved = ResolvedModelPaths {
            model_path: path.into(),
            context_limit: 0,
            settings: ModelSettings::default(),
            mmproj_path: std::env::var_os("GGUF_TEST_MMPROJ").map(Into::into),
            backend_id: None,
            draft_model_path: None,
        };
        let profile = profile(
            "test",
            &resolved,
            &llama_cpp_2::list_llama_ggml_backend_devices(),
        )
        .unwrap();
        assert!(profile
            .quantization
            .weights
            .contains_key(&Component::Decoder));
        println!("{}", serde_json::to_string_pretty(&profile).unwrap());
    }

    fn device(backend: &str, device_type: LlamaBackendDeviceType) -> LlamaBackendDevice {
        LlamaBackendDevice {
            index: 0,
            name: backend.to_string(),
            description: String::new(),
            backend: backend.to_string(),
            memory_total: 0,
            memory_free: 0,
            device_type,
        }
    }

    #[test]
    fn the_accelerator_is_the_first_gpu_unless_no_layer_is_offloaded() {
        let devices = [
            device("CPU", LlamaBackendDeviceType::Cpu),
            device("CUDA", LlamaBackendDeviceType::Gpu),
        ];
        assert_eq!(accelerator(Some(99), &devices), Accelerator::Cuda);
        assert_eq!(accelerator(None, &devices), Accelerator::Cuda);
        assert_eq!(accelerator(Some(0), &devices), Accelerator::Cpu);
        let mac = [device("MTL", LlamaBackendDeviceType::Gpu)];
        assert_eq!(accelerator(Some(99), &mac), Accelerator::Metal);
        assert_eq!(accelerator(Some(99), &devices[..1]), Accelerator::Cpu);
    }
}
