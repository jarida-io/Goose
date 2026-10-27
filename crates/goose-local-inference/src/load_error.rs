//! Local-model failures a host has to tell apart from every other engine error.
//!
//! A failure crosses the provider boundary as [`ProviderError::ExecutionError`], whose payload is
//! a string, and goose's agent loop turns it into reply prose ("Ran into this error: ..."). The
//! payload is this module's rendering, and [`ModelLoadError::from_provider_error`] reads it back,
//! so a host recognises the failure by type instead of by llama.cpp's log wording, which never
//! reaches it: llama.cpp reports an allocation that failed only as a null model or context.

use std::fmt;

use goose_provider_types::errors::ProviderError;

/// Which allocation ran out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadStage {
    /// The model's weights, while loading the file.
    Weights,
    /// A context: its KV cache and compute buffers, created on a model already loaded.
    Context,
}

impl LoadStage {
    fn as_str(self) -> &'static str {
        match self {
            Self::Weights => "model weights",
            Self::Context => "context",
        }
    }
}

/// A local model that could not load or start for a reason the household can act on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelLoadError {
    /// The GPU allocation failed and free memory was below what the stage needs. On a unified
    /// memory board (a Jetson) the GPU allocates from system memory, and `free_mb` is the
    /// kernel's `MemFree` after this process released its model files from the page cache.
    OutOfGpuMemory {
        stage: LoadStage,
        /// About what the stage needed, MiB: the file's size for weights, the KV cache plus
        /// reserves for a context.
        needed_mb: u64,
        free_mb: u64,
    },
}

const OUT_OF_GPU_MEMORY: &str = "out of GPU memory for the ";
const NEEDS: &str = ": needs about ";
const MIB_COMMA: &str = " MiB, ";
const MIB_FREE: &str = " MiB free";

impl fmt::Display for ModelLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfGpuMemory {
                stage,
                needed_mb,
                free_mb,
            } => write!(
                f,
                "{OUT_OF_GPU_MEMORY}{}{NEEDS}{needed_mb}{MIB_COMMA}{free_mb}{MIB_FREE}",
                stage.as_str()
            ),
        }
    }
}

impl std::error::Error for ModelLoadError {}

impl From<ModelLoadError> for ProviderError {
    fn from(error: ModelLoadError) -> Self {
        ProviderError::ExecutionError(error.to_string())
    }
}

impl ModelLoadError {
    /// The typed error a [`ProviderError`] carries, or `None` for any other failure.
    pub fn from_provider_error(error: &ProviderError) -> Option<Self> {
        match error {
            ProviderError::ExecutionError(text) => Self::parse(text),
            _ => None,
        }
    }

    fn parse(text: &str) -> Option<Self> {
        let rest = text.strip_prefix(OUT_OF_GPU_MEMORY)?;
        let (stage, rest) = [LoadStage::Weights, LoadStage::Context]
            .into_iter()
            .find_map(|stage| {
                rest.strip_prefix(stage.as_str())
                    .and_then(|r| r.strip_prefix(NEEDS))
                    .map(|r| (stage, r))
            })?;
        let (needed, rest) = rest.split_once(MIB_COMMA)?;
        let free = rest.strip_suffix(MIB_FREE)?;
        Some(Self::OutOfGpuMemory {
            stage,
            needed_mb: needed.parse().ok()?,
            free_mb: free.parse().ok()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_out_of_gpu_memory_error_survives_the_provider_boundary() {
        for stage in [LoadStage::Weights, LoadStage::Context] {
            let error = ModelLoadError::OutOfGpuMemory {
                stage,
                needed_mb: 3_052,
                free_mb: 251,
            };
            let crossed = ProviderError::from(error);
            assert!(matches!(crossed, ProviderError::ExecutionError(_)));
            assert_eq!(ModelLoadError::from_provider_error(&crossed), Some(error));
        }
    }

    #[test]
    fn the_rendering_names_the_stage_and_both_figures() {
        let error = ModelLoadError::OutOfGpuMemory {
            stage: LoadStage::Context,
            needed_mb: 1_664,
            free_mb: 251,
        };
        assert_eq!(
            error.to_string(),
            "out of GPU memory for the context: needs about 1664 MiB, 251 MiB free"
        );
    }

    #[test]
    fn other_engine_errors_are_not_mistaken_for_it() {
        for error in [
            ProviderError::ExecutionError("null result from llama cpp".to_string()),
            ProviderError::ExecutionError(
                "Failed to create context: null reference from llama.cpp".to_string(),
            ),
            ProviderError::ExecutionError(
                "out of GPU memory for the context: needs about many MiB, 1 MiB free".to_string(),
            ),
            ProviderError::ContextLengthExceeded(
                "out of GPU memory for the context: needs about 1 MiB, 1 MiB free".to_string(),
            ),
        ] {
            assert_eq!(ModelLoadError::from_provider_error(&error), None, "{error}");
        }
    }
}
