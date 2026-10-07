use std::{collections::HashMap, time::Duration};

use anyhow::{Ok, Result};
use opentelemetry_otlp::{
    MetricExporter, SpanExporter, WithExportConfig, WithHttpConfig, WithTonicConfig,
};
use opentelemetry_sdk::{
    metrics::{MeterProviderBuilder, SdkMeterProvider},
    trace::{Sampler, SdkTracerProvider, TracerProviderBuilder},
};
use serde::{Deserialize, Serialize};
use tonic::metadata::{MetadataKey, MetadataMap};
use url::Url;

use crate::auth::SecretString;

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct TelemetryConfig {
    #[serde(default)]
    pub traces: TracesConfig,
    #[serde(default)]
    pub metrics: MetricsConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct TracesConfig {
    pub enabled: bool,
    #[serde(default)]
    pub sampling: SamplingConfig,
    #[serde(default)]
    pub exporters: Vec<ExporterConfig>,
}

impl TracesConfig {
    /// Creates initializes the tracer provider builder with sampling & exporters
    /// according to the config.
    ///
    /// # Errors
    ///
    /// This will error if there is a failure creating the span exporters
    pub async fn tracer_provider_builder(&self) -> Result<TracerProviderBuilder> {
        let mut builder = SdkTracerProvider::builder().with_sampler(self.sampling.to_sampler());

        // add exporters
        for exporter_cfg in &self.exporters {
            let span_exporter = exporter_cfg.span_exporter().await?;
            builder = builder.with_batch_exporter(span_exporter);
        }

        Ok(builder)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SamplingConfig {
    #[serde(default = "crate::defaults::default_sampling_strategy")]
    pub strategy: SamplingStrategy,
    /// Sampling rate between 0.0 and 1.0 (used for Probabilistic strategy)
    #[serde(default = "crate::defaults::default_sampling_rate")]
    pub rate: f64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            strategy: crate::defaults::default_sampling_strategy(),
            rate: crate::defaults::default_sampling_rate(),
        }
    }
}

impl SamplingConfig {
    fn to_sampler(&self) -> opentelemetry_sdk::trace::Sampler {
        match self.strategy {
            SamplingStrategy::Always => Sampler::AlwaysOn,
            SamplingStrategy::Never => Sampler::AlwaysOff,
            SamplingStrategy::Probabilistic => Sampler::TraceIdRatioBased(self.rate),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SamplingStrategy {
    /// Always sample all spans
    Always,
    /// Never sample any spans
    Never,
    /// Sample spans based on probability (0.0 to 1.0)
    Probabilistic,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Protocol {
    #[serde(rename = "http")]
    Http,
    #[serde(rename = "grpc")]
    Grpc,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AuthConfig {
    Bearer {
        token: SecretString,
    },
    Basic {
        username: SecretString,
        password: SecretString,
    },
    Headers {
        headers: std::collections::HashMap<String, SecretString>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExporterConfig {
    pub name: String,
    pub url: Url,
    pub protocol: Protocol,
    #[serde(default = "crate::defaults::default_timeout_ms")]
    pub timeout: u64,
    #[serde(default)]
    pub auth: Option<AuthConfig>,
}

impl ExporterConfig {
    fn timeout_duration(&self) -> Duration {
        Duration::from_millis(self.timeout)
    }

    async fn auth_headers(&self) -> Result<HashMap<String, String>> {
        let mut headers = std::collections::HashMap::new();
        if let Some(auth) = &self.auth {
            match auth {
                AuthConfig::Bearer { token } => {
                    headers.insert(
                        "Authorization".to_string(),
                        format!("Bearer {}", token.resolve().await?),
                    );
                }
                AuthConfig::Basic { username, password } => {
                    use base64::{Engine as _, engine::general_purpose};
                    let credentials = format!(
                        "{}:{}",
                        username.resolve().await?,
                        password.resolve().await?
                    );
                    let encoded = general_purpose::STANDARD.encode(credentials);
                    headers.insert("Authorization".to_string(), format!("Basic {encoded}"));
                }
                AuthConfig::Headers {
                    headers: custom_headers,
                } => {
                    for (name, val) in custom_headers {
                        headers.insert(name.clone(), val.resolve().await?);
                    }
                }
            }
        }

        Ok(headers)
    }

    /// Creates a span exporter according to the config
    ///
    /// # Errors
    ///
    /// This will error if the config fails populating the
    /// authentication data via `SecretString`
    pub async fn span_exporter(&self) -> Result<SpanExporter> {
        let endpoint = self.url.to_string();
        let timeout = self.timeout_duration();
        let headers = self.auth_headers().await?;

        let exporter = match self.protocol {
            Protocol::Http => {
                let mut builder = SpanExporter::builder()
                    .with_http()
                    .with_endpoint(endpoint)
                    .with_timeout(timeout);

                // Add headers if any
                if !headers.is_empty() {
                    builder = builder.with_headers(headers);
                }

                builder.build()?
            }
            Protocol::Grpc => {
                let mut builder = SpanExporter::builder()
                    .with_tonic()
                    .with_endpoint(endpoint)
                    .with_timeout(timeout);

                if !headers.is_empty() {
                    // Convert HashMap to MetadataMap
                    let mut metadata = MetadataMap::new();
                    for (key, value) in headers {
                        metadata.insert(MetadataKey::from_bytes(key.as_bytes())?, value.parse()?);
                    }
                    builder = builder.with_metadata(metadata);
                }

                builder.build()?
            }
        };

        Ok(exporter)
    }

    /// Creates a metrics exporter according to the config
    ///
    /// # Errors
    ///
    /// This will error if the config fails populating the
    /// authentication data via `SecretString`
    pub async fn metrics_exporter(&self) -> Result<MetricExporter> {
        let endpoint = self.url.to_string();
        let timeout = self.timeout_duration();
        let headers = self.auth_headers().await?;

        let exporter = match &self.protocol {
            Protocol::Http => {
                let mut builder = MetricExporter::builder()
                    .with_http()
                    .with_endpoint(endpoint)
                    .with_timeout(timeout);

                if !headers.is_empty() {
                    builder = builder.with_headers(headers);
                }

                builder.build()?
            }
            Protocol::Grpc => {
                let mut builder = MetricExporter::builder()
                    .with_tonic()
                    .with_endpoint(endpoint)
                    .with_timeout(timeout);

                if !headers.is_empty() {
                    // Convert HashMap to MetadataMap
                    let mut metadata = MetadataMap::new();
                    for (key, value) in headers {
                        metadata.insert(MetadataKey::from_bytes(key.as_bytes())?, value.parse()?);
                    }
                    builder = builder.with_metadata(metadata);
                }

                builder.build()?
            }
        };

        Ok(exporter)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct MetricsConfig {
    pub enabled: bool,
    #[serde(default)]
    pub exporters: Vec<ExporterConfig>,
}

impl MetricsConfig {
    /// Creates initializes the meter provider builder with sampling & exporters
    /// according to the config.
    ///
    /// # Errors
    ///
    /// This will error if there is a failure creating the span exporters
    pub async fn meter_provider_builder(&self) -> Result<MeterProviderBuilder> {
        let mut builder = SdkMeterProvider::builder();

        // add exporters
        for exporter_cfg in &self.exporters {
            let exporter = exporter_cfg.metrics_exporter().await?;
            builder = builder.with_periodic_exporter(exporter);
        }

        Ok(builder)
    }
}
