//! What one backend serving one model file can do. The model card is goose's [`CanonicalModel`];
//! [`Endpoint`] follows OpenRouter's per-endpoint fields, since OpenRouter's "a provider serving a
//! model" is a local backend serving a file. [`Quantization`] and [`Caches`] go below what either
//! schema describes, in TFLite/ONNX element-type names and Hugging Face's KV cache layouts.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::canonical::CanonicalModel;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProfile {
    pub model: CanonicalModel,
    pub endpoint: Endpoint,
    pub quantization: Quantization,
    pub caches: Caches,
    pub runtime: Runtime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Endpoint {
    /// OpenRouter's bucket for the decoder's weights: `int4`, `int8`, `fp4`, `fp8`, `fp16`,
    /// `bf16`, `fp32` or `unknown`.
    pub quantization: String,
    /// The window this backend will serve; `None` when it is sized from free memory at load.
    pub context_length: Option<usize>,
    pub max_completion_tokens: Option<usize>,
    /// OpenRouter parameter names this backend honours. Local backends read them from the
    /// model's settings, since goose's provider call carries none of them.
    pub supported_parameters: Vec<String>,
    pub supports_tool_choice: ToolChoiceSupport,
    /// Whether a request that repeats an earlier prompt's beginning reuses its cached state.
    pub supports_implicit_caching: bool,
}

/// OpenRouter's `supports_tool_choice`: which OpenAI `tool_choice` values are honoured.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolChoiceSupport {
    pub none: bool,
    pub auto: bool,
    pub required: bool,
    pub function: bool,
}

/// Element types by their TFLite/ONNX names. GGML's block formats fall into the integer type of
/// their width, with the block format kept in [`Precision::scheme`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Dtype {
    Fp32,
    Fp16,
    Bf16,
    Fp8,
    Fp4,
    Int8,
    Int6,
    Int5,
    Int4,
    Int3,
    Int2,
    Int1,
}

impl Dtype {
    /// The value OpenRouter's `quantization` field takes for weights of this type.
    pub fn openrouter_bucket(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Fp16 => "fp16",
            Self::Bf16 => "bf16",
            Self::Fp8 => "fp8",
            Self::Fp4 => "fp4",
            Self::Int8 => "int8",
            Self::Int4 => "int4",
            Self::Int6 | Self::Int5 | Self::Int3 | Self::Int2 | Self::Int1 => "unknown",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Precision {
    pub dtype: Dtype,
    /// The format's own name where it says more than the type: `Q4_K`, `q8_0`, `IQ4_XS`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scheme: Option<String>,
    /// Stored bits per value, block scales included: `Q8_0` is 8.5.
    pub bits_per_value: f32,
}

impl Precision {
    pub fn plain(dtype: Dtype, bits_per_value: f32) -> Self {
        Self {
            dtype,
            scheme: None,
            bits_per_value,
        }
    }

    pub fn with_scheme(dtype: Dtype, scheme: &str, bits_per_value: f32) -> Self {
        Self {
            dtype,
            scheme: Some(scheme.to_string()),
            bits_per_value,
        }
    }
}

/// The parts of a model whose weights can be stored differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Component {
    Decoder,
    TokenEmbeddings,
    PerLayerEmbeddings,
    VisionEncoder,
    AudioEncoder,
    Drafter,
    Other,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeightShare {
    pub precision: Precision,
    pub bytes: u64,
}

/// Who decides a value, and what else it could be.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Setting<T> {
    pub value: T,
    pub chosen_by: ChosenBy,
    /// The values the chooser may pick from; empty when only the model file decides.
    #[serde(default = "Vec::new", skip_serializing_if = "Vec::is_empty")]
    pub options: Vec<T>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChosenBy {
    /// Fixed when the model was converted.
    ModelFile,
    /// The engine picks it, possibly at load.
    Engine,
    /// The model's settings name it.
    Settings,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Quantization {
    /// Each component's weights by precision, largest share first.
    pub weights: BTreeMap<Component, Vec<WeightShare>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub activations: Option<Setting<Dtype>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_key: Option<Setting<Precision>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_value: Option<Setting<Precision>>,
}

impl Quantization {
    /// Counts `bytes` of `component`'s weights as stored in `precision`; each component's
    /// shares stay largest first.
    pub fn add_weights(&mut self, component: Component, precision: Precision, bytes: u64) {
        let shares = self.weights.entry(component).or_default();
        match shares.iter_mut().find(|share| share.precision == precision) {
            Some(share) => share.bytes += bytes,
            None => shares.push(WeightShare { precision, bytes }),
        }
        shares.sort_by(|a, b| b.bytes.cmp(&a.bytes));
    }

    /// The type holding most of the decoder's bytes.
    pub fn decoder_dtype(&self) -> Option<Dtype> {
        self.weights
            .get(&Component::Decoder)?
            .iter()
            .max_by_key(|share| share.bytes)
            .map(|share| share.precision.dtype)
    }

    pub fn openrouter_bucket(&self) -> &'static str {
        self.decoder_dtype()
            .map_or("unknown", Dtype::openrouter_bucket)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Caches {
    pub kv: KvCache,
    pub prefix: PrefixCache,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snapshot: Option<SnapshotCache>,
    /// Compiled GPU programs kept on disk between loads.
    pub compiled_programs: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvCache {
    pub layout: KvLayout,
    /// Positions the cache holds; `None` when sized from free memory at load.
    pub capacity_tokens: Option<usize>,
    /// Bytes each position costs on the layers that keep every position.
    pub bytes_per_token: Option<u64>,
    /// Bytes the sliding-window layers take whatever the window.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fixed_bytes: Option<u64>,
    pub rewind: Rewind,
}

/// Named after Hugging Face transformers' cache classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum KvLayout {
    /// Every layer keeps every position.
    Static,
    /// Sliding-window layers keep only their window; the rest keep every position.
    Hybrid {
        window: u64,
        sliding_layers: u64,
        full_layers: u64,
    },
}

/// How far back a cache can be rewound and still be reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Rewind {
    /// To any position.
    Free,
    /// Within the window behind its end, or to a checkpoint taken earlier.
    Checkpointed,
    /// Only appended to.
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefixCache {
    pub reuse: PrefixReuse,
    pub side_calls: SideCalls,
    /// Whether reuse continues through a picture or recording that has not changed.
    pub through_media: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefixReuse {
    /// Only a request that extends the cached conversation reuses it.
    AppendOnly,
    /// Any request reuses the longest beginning it shares with the cache.
    LongestPrefix,
}

/// What happens to the cached conversation while an unrelated request runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SideCalls {
    /// The request gets a context of its own; the conversation's stays.
    OwnContext,
    /// The conversation's cache is written to disk and restored afterwards.
    SetAsideToDisk,
    /// The engine copies the conversation's cache while the request runs.
    CopiedInMemory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotCache {
    pub holds: SnapshotContents,
    /// Whether a file can hold what someone said, not only the system prompt and tools.
    pub personal: bool,
    /// Files kept; `None` when kept by age or role instead of by count.
    pub kept: Option<usize>,
    /// Whether the files are readable by their owner only.
    pub owner_only: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SnapshotContents {
    /// The beginning every prompt for the model shares: system prompt and tools.
    StablePrefix,
    /// A whole conversation.
    WholeConversation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Runtime {
    /// The goose backend id: `llamacpp`, `litert` or `mlx`.
    pub backend: String,
    pub accelerator: Accelerator,
    pub speculative_decoding: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Accelerator {
    Cuda,
    Metal,
    WebGpu,
    Vulkan,
    Cpu,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn share(dtype: Dtype, bytes: u64) -> WeightShare {
        WeightShare {
            precision: Precision::plain(dtype, 4.0),
            bytes,
        }
    }

    #[test]
    fn the_bucket_is_the_decoders_largest_share() {
        let mut quantization = Quantization::default();
        quantization.weights.insert(
            Component::Decoder,
            vec![share(Dtype::Fp32, 10), share(Dtype::Int4, 900)],
        );
        quantization
            .weights
            .insert(Component::TokenEmbeddings, vec![share(Dtype::Int2, 5_000)]);
        assert_eq!(quantization.decoder_dtype(), Some(Dtype::Int4));
        assert_eq!(quantization.openrouter_bucket(), "int4");
    }

    #[test]
    fn the_same_precision_adds_up_and_the_largest_share_leads() {
        let mut quantization = Quantization::default();
        let q4 = Precision::with_scheme(Dtype::Int4, "Q4_0", 4.5);
        quantization.add_weights(Component::Decoder, Precision::plain(Dtype::Fp32, 32.0), 10);
        quantization.add_weights(Component::Decoder, q4.clone(), 400);
        quantization.add_weights(Component::Decoder, q4, 600);
        let decoder = &quantization.weights[&Component::Decoder];
        assert_eq!(decoder.len(), 2);
        assert_eq!(decoder[0].bytes, 1_000);
        assert_eq!(decoder[0].precision.dtype, Dtype::Int4);
    }

    #[test]
    fn widths_openrouter_does_not_name_are_unknown() {
        assert_eq!(Dtype::Int2.openrouter_bucket(), "unknown");
        assert_eq!(Quantization::default().openrouter_bucket(), "unknown");
    }

    #[test]
    fn layouts_and_components_serialize_as_plain_names() {
        let json = serde_json::to_value(KvLayout::Hybrid {
            window: 512,
            sliding_layers: 28,
            full_layers: 7,
        })
        .unwrap();
        assert_eq!(json["kind"], "hybrid");
        let mut weights = BTreeMap::new();
        weights.insert(Component::PerLayerEmbeddings, Vec::<WeightShare>::new());
        let json = serde_json::to_value(weights).unwrap();
        assert!(json.get("per_layer_embeddings").is_some());
    }
}
