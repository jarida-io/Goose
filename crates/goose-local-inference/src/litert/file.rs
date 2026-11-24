//! What a `.litertlm` file stores, read from its section table and its TFLite graphs' tables
//! without touching weight data: each component's element types and the decode signature's
//! KV cache inputs. LiteRT-LM's C API reports neither.
//!
//! Layout: `schema/core/litertlm_header_schema.fbs` (LiteRT-LM) and TFLite's `schema.fbs`.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use goose_provider_types::model_profile::{Component, Dtype, Precision, Quantization};

const MAGIC: &[u8; 8] = b"LITERTLM";
/// `AnySectionDataType::TFLiteModel`.
const SECTION_TFLITE_MODEL: u8 = 3;
/// `VData::StringValue`.
const VALUE_STRING: u8 = 9;
const BLOCK: u64 = 4096;

/// A file's section-level facts.
#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct LitertlmFile {
    /// Each component's weights; nothing else in it is filled.
    pub quantization: Quantization,
    /// The decode signature's KV inputs; empty when the file has none.
    pub kv: Vec<KvInput>,
    /// The text decoder's `prefer_activation_type`, when the file names one.
    pub prefer_activation: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct KvInput {
    pub name: String,
    pub dtype: Dtype,
    pub bits_per_value: f32,
    pub shape: Vec<i64>,
}

impl KvInput {
    pub fn is_key(&self) -> bool {
        self.name.starts_with("kv_cache_k_")
    }

    /// The sequence axis is the longest: the model's context placeholder, resized at load.
    pub fn positions(&self) -> u64 {
        self.shape.iter().copied().max().unwrap_or(0).max(0) as u64
    }

    /// Bytes one cached position costs in this input.
    pub fn bytes_per_position(&self) -> u64 {
        let elements: i64 = self.shape.iter().product();
        let per_position = (elements.max(0) as u64)
            .checked_div(self.positions())
            .unwrap_or(0);
        (per_position as f64 * f64::from(self.bits_per_value) / 8.0).ceil() as u64
    }
}

pub(super) fn read(path: &Path) -> Result<LitertlmFile, String> {
    let mut source = Source::open(path)?;
    if source.bytes(0, 8)? != MAGIC {
        return Err(format!("{} is not a LiteRT-LM file", path.display()));
    }
    let header_end = source.u64(24)?;
    let header = Fb::new(&mut source, 32, header_end);
    let sections = header.sections()?;

    let mut file = LitertlmFile::default();
    for section in sections
        .iter()
        .filter(|s| s.data_type == SECTION_TFLITE_MODEL)
    {
        let model_type = section.items.get("model_type").map(String::as_str);
        let Some(component) = model_type.and_then(component_of) else {
            continue;
        };
        if component == Component::Decoder {
            file.prefer_activation = section.items.get("prefer_activation_type").cloned();
        }
        let mut graph = Fb::new(&mut source, section.begin, section.end);
        let (constants, kv) = graph.tflite(component == Component::Decoder)?;
        for (precision, bytes) in constants {
            file.quantization.add_weights(component, precision, bytes);
        }
        if component == Component::Decoder {
            file.kv = kv;
        }
    }
    Ok(file)
}

/// The component a section's `model_type` stores; `None` for marker sections that hold no
/// weights to speak of.
fn component_of(model_type: &str) -> Option<Component> {
    Some(match model_type {
        "tf_lite_prefill_decode" => Component::Decoder,
        "tf_lite_embedder" => Component::TokenEmbeddings,
        "tf_lite_per_layer_embedder" => Component::PerLayerEmbeddings,
        "tf_lite_vision_encoder" | "tf_lite_vision_adapter" => Component::VisionEncoder,
        "tf_lite_audio_encoder_hw" | "tf_lite_audio_adapter" => Component::AudioEncoder,
        "tf_lite_mtp_drafter" => Component::Drafter,
        t if t.starts_with("tf_lite_end_of_") => return None,
        _ => Component::Other,
    })
}

/// A TFLite `TensorType` as weight precision; `None` for index and bookkeeping types.
fn tflite_precision(tensor_type: u8) -> Option<Precision> {
    let (dtype, bits) = match tensor_type {
        0 => (Dtype::Fp32, 32.0),
        1 => (Dtype::Fp16, 16.0),
        18 => (Dtype::Bf16, 16.0),
        3 | 9 => (Dtype::Int8, 8.0),
        17 | 20 => (Dtype::Int4, 4.0),
        19 => (Dtype::Int2, 2.0),
        21 | 22 => (Dtype::Fp8, 8.0),
        _ => return None,
    };
    Some(Precision::plain(dtype, bits))
}

struct Section {
    begin: u64,
    end: u64,
    data_type: u8,
    items: BTreeMap<String, String>,
}

/// Random access to the file in 4 KiB blocks, each read once.
struct Source {
    file: File,
    len: u64,
    blocks: BTreeMap<u64, Vec<u8>>,
}

impl Source {
    fn open(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let len = file.metadata().map_err(|e| e.to_string())?.len();
        Ok(Self {
            file,
            len,
            blocks: BTreeMap::new(),
        })
    }

    fn block(&mut self, index: u64) -> Result<&[u8], String> {
        if !self.blocks.contains_key(&index) {
            let start = index * BLOCK;
            let size = BLOCK.min(self.len.saturating_sub(start)) as usize;
            let mut buf = vec![0; size];
            read_exact_at(&self.file, &mut buf, start).map_err(|e| e.to_string())?;
            self.blocks.insert(index, buf);
        }
        Ok(&self.blocks[&index])
    }

    fn bytes(&mut self, offset: u64, len: usize) -> Result<Vec<u8>, String> {
        let end = offset
            .checked_add(len as u64)
            .filter(|&end| end <= self.len)
            .ok_or_else(|| format!("read past the end at {offset}"))?;
        let mut out = Vec::with_capacity(len);
        let mut at = offset;
        while at < end {
            let block = self.block(at / BLOCK)?;
            let from = (at % BLOCK) as usize;
            let take = (block.len() - from).min((end - at) as usize);
            out.extend_from_slice(&block[from..from + take]);
            at += take as u64;
        }
        Ok(out)
    }

    fn u64(&mut self, offset: u64) -> Result<u64, String> {
        Ok(u64::from_le_bytes(
            self.bytes(offset, 8)?.try_into().expect("8 bytes"),
        ))
    }
}

#[cfg(unix)]
fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
}

#[cfg(windows)]
fn read_exact_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        let n = file.seek_read(buf, offset)?;
        if n == 0 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        buf = &mut buf[n..];
        offset += n as u64;
    }
    Ok(())
}

/// A flatbuffer occupying `[base, end)` of the file; offsets are relative to `base`.
struct Fb<'a> {
    source: &'a mut Source,
    base: u64,
    end: u64,
}

impl<'a> Fb<'a> {
    fn new(source: &'a mut Source, base: u64, end: u64) -> Self {
        Self { source, base, end }
    }

    fn bytes(&mut self, at: u64, len: usize) -> Result<Vec<u8>, String> {
        if at
            .checked_add(len as u64)
            .is_none_or(|e| e > self.end - self.base)
        {
            return Err(format!("flatbuffer read out of bounds at {at}"));
        }
        self.source.bytes(self.base + at, len)
    }

    fn u8(&mut self, at: u64) -> Result<u8, String> {
        Ok(self.bytes(at, 1)?[0])
    }

    fn u16(&mut self, at: u64) -> Result<u16, String> {
        Ok(u16::from_le_bytes(
            self.bytes(at, 2)?.try_into().expect("2"),
        ))
    }

    fn u32(&mut self, at: u64) -> Result<u32, String> {
        Ok(u32::from_le_bytes(
            self.bytes(at, 4)?.try_into().expect("4"),
        ))
    }

    fn i32(&mut self, at: u64) -> Result<i32, String> {
        Ok(i32::from_le_bytes(
            self.bytes(at, 4)?.try_into().expect("4"),
        ))
    }

    fn u64(&mut self, at: u64) -> Result<u64, String> {
        Ok(u64::from_le_bytes(
            self.bytes(at, 8)?.try_into().expect("8"),
        ))
    }

    fn deref(&mut self, at: u64) -> Result<u64, String> {
        Ok(at + u64::from(self.u32(at)?))
    }

    /// Where field `index` of the table at `table` is stored, `None` when absent.
    fn field(&mut self, table: u64, index: u16) -> Result<Option<u64>, String> {
        let vtable = (table as i64 - i64::from(self.i32(table)?)) as u64;
        let slot = 4 + 2 * u64::from(index);
        if slot >= u64::from(self.u16(vtable)?) {
            return Ok(None);
        }
        let offset = self.u16(vtable + slot)?;
        Ok((offset != 0).then_some(table + u64::from(offset)))
    }

    fn table_field(&mut self, table: u64, index: u16) -> Result<Option<u64>, String> {
        match self.field(table, index)? {
            Some(at) => Ok(Some(self.deref(at)?)),
            None => Ok(None),
        }
    }

    fn string(&mut self, at: u64) -> Result<String, String> {
        let s = self.deref(at)?;
        let len = self.u32(s)? as usize;
        Ok(String::from_utf8_lossy(&self.bytes(s + 4, len)?).into_owned())
    }

    /// The positions of a vector's elements: tables (dereference each) or scalars.
    fn vector(&mut self, at: u64, width: u64) -> Result<Vec<u64>, String> {
        let v = self.deref(at)?;
        let n = u64::from(self.u32(v)?);
        Ok((0..n).map(|i| v + 4 + width * i).collect())
    }

    fn tables(&mut self, at: Option<u64>) -> Result<Vec<u64>, String> {
        let Some(at) = at else {
            return Ok(Vec::new());
        };
        self.vector(at, 4)?
            .into_iter()
            .map(|p| self.deref(p))
            .collect()
    }

    fn ints(&mut self, at: Option<u64>) -> Result<Vec<i32>, String> {
        let Some(at) = at else {
            return Ok(Vec::new());
        };
        self.vector(at, 4)?
            .into_iter()
            .map(|p| self.i32(p))
            .collect()
    }

    fn sections(mut self) -> Result<Vec<Section>, String> {
        let root = self.deref(0)?;
        let Some(metadata) = self.table_field(root, 1)? else {
            return Ok(Vec::new());
        };
        let objects = self.field(metadata, 0)?;
        let mut sections = Vec::new();
        for object in self.tables(objects)? {
            let begin = match self.field(object, 1)? {
                Some(at) => self.u64(at)?,
                None => 0,
            };
            let end = match self.field(object, 2)? {
                Some(at) => self.u64(at)?,
                None => 0,
            };
            let data_type = match self.field(object, 3)? {
                Some(at) => self.u8(at)?,
                None => 0,
            };
            let mut items = BTreeMap::new();
            let pairs = self.field(object, 0)?;
            for pair in self.tables(pairs)? {
                let Some(key) = self.field(pair, 0)? else {
                    continue;
                };
                let value_type = match self.field(pair, 1)? {
                    Some(at) => self.u8(at)?,
                    None => 0,
                };
                if value_type != VALUE_STRING {
                    continue;
                }
                let Some(value) = self.table_field(pair, 2)? else {
                    continue;
                };
                let Some(text) = self.field(value, 0)? else {
                    continue;
                };
                let key = self.string(key)?;
                items.insert(key, self.string(text)?);
            }
            sections.push(Section {
                begin,
                end,
                data_type,
                items,
            });
        }
        Ok(sections)
    }

    /// Subgraph 0's constant tensors as (precision, bytes), and, when asked, the decode
    /// signature's KV inputs. A constant is a tensor no operator writes and the graph does not
    /// take in; its bytes follow from its shape, since the weight data is never read.
    #[allow(clippy::type_complexity)]
    fn tflite(&mut self, want_kv: bool) -> Result<(Vec<(Precision, u64)>, Vec<KvInput>), String> {
        let model = self.deref(0)?;
        let subgraphs = self.field(model, 2)?;
        let subgraphs = self.tables(subgraphs)?;
        let Some(&graph) = subgraphs.first() else {
            return Ok((Vec::new(), Vec::new()));
        };
        let tensors_at = self.field(graph, 0)?;
        let tensors = self.tables(tensors_at)?;
        let mut produced: std::collections::BTreeSet<i32> = {
            let inputs = self.field(graph, 1)?;
            self.ints(inputs)?.into_iter().collect()
        };
        let operators = self.field(graph, 3)?;
        for op in self.tables(operators)? {
            let outputs = self.field(op, 2)?;
            produced.extend(self.ints(outputs)?);
        }

        let mut constants = Vec::new();
        for (index, &tensor) in tensors.iter().enumerate() {
            if produced.contains(&(index as i32)) {
                continue;
            }
            let (tensor_type, shape) = self.tensor(tensor)?;
            let Some(precision) = tflite_precision(tensor_type) else {
                continue;
            };
            let elements: i64 = shape.iter().map(|&d| d.max(1)).product();
            let bytes = (elements as f64 * f64::from(precision.bits_per_value) / 8.0).ceil();
            constants.push((precision, bytes as u64));
        }

        let kv = if want_kv {
            self.decode_kv(model, &subgraphs)?
        } else {
            Vec::new()
        };
        Ok((constants, kv))
    }

    fn tensor(&mut self, tensor: u64) -> Result<(u8, Vec<i64>), String> {
        let tensor_type = match self.field(tensor, 1)? {
            Some(at) => self.u8(at)?,
            None => 0,
        };
        let shape_at = self.field(tensor, 0)?;
        let shape = self.ints(shape_at)?.into_iter().map(i64::from).collect();
        Ok((tensor_type, shape))
    }

    fn decode_kv(&mut self, model: u64, subgraphs: &[u64]) -> Result<Vec<KvInput>, String> {
        let signatures = self.field(model, 7)?;
        for signature in self.tables(signatures)? {
            let Some(key) = self.field(signature, 2)? else {
                continue;
            };
            if self.string(key)? != "decode" {
                continue;
            }
            let graph_index = match self.field(signature, 4)? {
                Some(at) => self.u32(at)? as usize,
                None => 0,
            };
            let Some(&graph) = subgraphs.get(graph_index) else {
                return Ok(Vec::new());
            };
            let tensors_at = self.field(graph, 0)?;
            let tensors = self.tables(tensors_at)?;
            let inputs = self.field(signature, 0)?;
            let mut kv = Vec::new();
            for map in self.tables(inputs)? {
                let Some(name_at) = self.field(map, 0)? else {
                    continue;
                };
                let name = self.string(name_at)?;
                if !name.starts_with("kv_cache_") {
                    continue;
                }
                let index = match self.field(map, 1)? {
                    Some(at) => self.u32(at)? as usize,
                    None => 0,
                };
                let Some(&tensor) = tensors.get(index) else {
                    continue;
                };
                let (tensor_type, shape) = self.tensor(tensor)?;
                let Some(precision) = tflite_precision(tensor_type) else {
                    continue;
                };
                kv.push(KvInput {
                    name,
                    dtype: precision.dtype,
                    bits_per_value: precision.bits_per_value,
                    shape,
                });
            }
            return Ok(kv);
        }
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sections_map_to_the_components_they_store() {
        assert_eq!(
            component_of("tf_lite_prefill_decode"),
            Some(Component::Decoder)
        );
        assert_eq!(
            component_of("tf_lite_per_layer_embedder"),
            Some(Component::PerLayerEmbeddings)
        );
        assert_eq!(
            component_of("tf_lite_audio_encoder_hw"),
            Some(Component::AudioEncoder)
        );
        assert_eq!(
            component_of("tf_lite_mtp_drafter"),
            Some(Component::Drafter)
        );
        assert_eq!(component_of("tf_lite_end_of_vision"), None);
        assert_eq!(
            component_of("tf_lite_something_new"),
            Some(Component::Other)
        );
    }

    #[test]
    fn tflite_types_name_their_width() {
        assert_eq!(tflite_precision(19).unwrap().dtype, Dtype::Int2);
        assert_eq!(tflite_precision(17).unwrap().bits_per_value, 4.0);
        assert_eq!(tflite_precision(9).unwrap().dtype, Dtype::Int8);
        assert!(
            tflite_precision(2).is_none(),
            "int32 holds indices, not weights"
        );
    }

    #[test]
    fn a_kv_input_costs_its_width_per_position() {
        let key = KvInput {
            name: "kv_cache_k_0".to_string(),
            dtype: Dtype::Int8,
            bits_per_value: 8.0,
            shape: vec![1, 2, 32003, 256],
        };
        assert!(key.is_key());
        assert_eq!(key.positions(), 32003);
        assert_eq!(key.bytes_per_position(), 512);
        let value = KvInput {
            name: "kv_cache_v_0".to_string(),
            shape: vec![1, 2, 256, 32003],
            ..key
        };
        assert!(!value.is_key());
        assert_eq!(value.bytes_per_position(), 512);
    }

    /// Reads a real file when `LITERTLM_TEST_FILE` names one, such as LiteRT-LM's
    /// `runtime/testdata/fake_gemma4.litertlm`.
    #[test]
    fn reads_a_real_file_when_given_one() {
        let Some(path) = std::env::var_os("LITERTLM_TEST_FILE") else {
            return;
        };
        let file = read(Path::new(&path)).unwrap();
        assert!(
            file.quantization.weights.contains_key(&Component::Decoder),
            "{file:#?}"
        );
        println!("{file:#?}");
    }
}
