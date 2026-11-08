//! Single-tower copies of a multimodal projector. mtmd loads every tower a file declares, and a
//! Gemma 4 encoder's picture tower is over a third of it: a turn that only brings sound need not
//! hold it, nor one that only brings pictures the audio tower.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Tower {
    Audio,
    Vision,
}

impl Tower {
    fn flag(self) -> &'static str {
        match self {
            Tower::Audio => "clip.has_audio_encoder",
            Tower::Vision => "clip.has_vision_encoder",
        }
    }

    fn other(self) -> Tower {
        match self {
            Tower::Audio => Tower::Vision,
            Tower::Vision => Tower::Audio,
        }
    }

    /// The vision projector's `mm.input_projection` has no tower prefix, so an audio copy keeps
    /// it; nothing loads it there.
    fn owns(self, tensor: &str) -> bool {
        match self {
            Tower::Audio => tensor.starts_with("a.") || tensor.starts_with("mm.a."),
            Tower::Vision => tensor.starts_with("v."),
        }
    }

    pub(super) fn name(self) -> &'static str {
        match self {
            Tower::Audio => "audio",
            Tower::Vision => "vision",
        }
    }
}

struct KvEntry {
    key: String,
    span: (u64, u64),
    flag: Option<bool>,
}

struct TensorEntry {
    name: String,
    /// The entry less its trailing offset field.
    span: (u64, u64),
    offset: u64,
}

struct Header {
    kv: Vec<KvEntry>,
    tensors: Vec<TensorEntry>,
    alignment: u64,
    end: u64,
    data_start: u64,
    file_len: u64,
}

struct Counted<R> {
    inner: R,
    pos: u64,
}

impl<R: Read> Counted<R> {
    fn bytes<const N: usize>(&mut self) -> std::io::Result<[u8; N]> {
        let mut buf = [0u8; N];
        self.inner.read_exact(&mut buf)?;
        self.pos += N as u64;
        Ok(buf)
    }

    fn u32(&mut self) -> std::io::Result<u32> {
        self.bytes::<4>().map(u32::from_le_bytes)
    }

    fn u64(&mut self) -> std::io::Result<u64> {
        self.bytes::<8>().map(u64::from_le_bytes)
    }

    fn string(&mut self) -> std::io::Result<String> {
        let len = self.u64()?;
        if len > 1 << 20 {
            return Err(invalid("a GGUF string longer than 1 MiB"));
        }
        let mut buf = vec![0u8; len as usize];
        self.inner.read_exact(&mut buf)?;
        self.pos += len;
        String::from_utf8(buf).map_err(|_| invalid("a GGUF string that is not UTF-8"))
    }

    fn skip(&mut self, n: u64) -> std::io::Result<()> {
        let copied = std::io::copy(&mut (&mut self.inner).take(n), &mut std::io::sink())?;
        if copied != n {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        self.pos += n;
        Ok(())
    }

    fn skip_value(&mut self, ty: u32) -> std::io::Result<()> {
        match ty {
            GGUF_TYPE_STRING => self.string().map(drop),
            GGUF_TYPE_ARRAY => {
                let item = self.u32()?;
                let n = self.u64()?;
                match scalar_size(item) {
                    Some(size) => self.skip(size * n),
                    None => (0..n).try_for_each(|_| self.skip_value(item)),
                }
            }
            _ => self.skip(scalar_size(ty).ok_or_else(|| invalid("an unknown GGUF value type"))?),
        }
    }
}

fn scalar_size(ty: u32) -> Option<u64> {
    match ty {
        0 | 1 | GGUF_TYPE_BOOL => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

fn invalid(what: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, what.to_string())
}

fn align_up(n: u64, alignment: u64) -> u64 {
    n.div_ceil(alignment) * alignment
}

fn read_header(path: &Path) -> std::io::Result<Header> {
    let file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut r = Counted {
        inner: BufReader::new(file),
        pos: 0,
    };
    if &r.bytes::<4>()? != GGUF_MAGIC {
        return Err(invalid("not a GGUF file"));
    }
    if r.u32()? < 2 {
        return Err(invalid("a GGUF version before 2"));
    }
    let n_tensors = r.u64()?;
    let n_kv = r.u64()?;
    let mut alignment = DEFAULT_ALIGNMENT;
    let mut kv = Vec::with_capacity(n_kv.min(4096) as usize);
    for _ in 0..n_kv {
        let start = r.pos;
        let key = r.string()?;
        let ty = r.u32()?;
        let mut flag = None;
        match ty {
            GGUF_TYPE_BOOL => flag = Some(r.bytes::<1>()?[0] != 0),
            GGUF_TYPE_UINT32 if key == "general.alignment" => alignment = u64::from(r.u32()?),
            _ => r.skip_value(ty)?,
        }
        kv.push(KvEntry {
            key,
            span: (start, r.pos),
            flag,
        });
    }
    if alignment == 0 {
        return Err(invalid("a GGUF alignment of 0"));
    }
    let mut tensors = Vec::with_capacity(n_tensors.min(1 << 16) as usize);
    for _ in 0..n_tensors {
        let start = r.pos;
        let name = r.string()?;
        let n_dims = r.u32()?;
        r.skip(u64::from(n_dims) * 8 + 4)?;
        let span = (start, r.pos);
        let offset = r.u64()?;
        tensors.push(TensorEntry { name, span, offset });
    }
    let end = r.pos;
    Ok(Header {
        kv,
        tensors,
        alignment,
        end,
        data_start: align_up(end, alignment),
        file_len,
    })
}

/// What a copy keeping `keep` holds: the source's entries to keep and each kept tensor's bytes.
struct Plan {
    header: Header,
    kept: Vec<usize>,
    /// Bytes of each tensor in the source, padding included, by index.
    regions: Vec<u64>,
}

impl Plan {
    fn data_len(&self) -> u64 {
        self.kept.iter().map(|&i| self.regions[i]).sum()
    }

    fn unpadded_header_len(&self, keep: Tower) -> u64 {
        let other = keep.other().flag();
        let kv: u64 = self
            .header
            .kv
            .iter()
            .map(|e| {
                if e.key == other {
                    8 + e.key.len() as u64 + 4 + 1
                } else {
                    e.span.1 - e.span.0
                }
            })
            .sum();
        let tensors: u64 = self
            .kept
            .iter()
            .map(|&i| {
                let t = &self.header.tensors[i];
                t.span.1 - t.span.0 + 8
            })
            .sum();
        4 + 4 + 8 + 8 + kv + tensors
    }

    fn header_len(&self, keep: Tower) -> u64 {
        align_up(self.unpadded_header_len(keep), self.header.alignment)
    }
}

/// `None` when the file is not a two-tower projector, so a copy would hold everything anyway.
fn plan(source: &Path, keep: Tower) -> std::io::Result<Option<Plan>> {
    let header = read_header(source)?;
    let flag = |tower: Tower| {
        header
            .kv
            .iter()
            .find(|e| e.key == tower.flag())
            .and_then(|e| e.flag)
    };
    if flag(keep) != Some(true) || flag(keep.other()) != Some(true) {
        return Ok(None);
    }
    if header
        .tensors
        .windows(2)
        .any(|w| w[0].offset >= w[1].offset)
    {
        return Err(invalid("GGUF tensors out of offset order"));
    }
    let data_len = header
        .file_len
        .checked_sub(header.data_start)
        .ok_or_else(|| invalid("a GGUF file shorter than its header"))?;
    let regions = (0..header.tensors.len())
        .map(|i| {
            let next = header.tensors.get(i + 1).map_or(data_len, |t| t.offset);
            next.checked_sub(header.tensors[i].offset)
                .ok_or_else(|| invalid("a GGUF tensor past the end of the file"))
        })
        .collect::<std::io::Result<Vec<u64>>>()?;
    let kept: Vec<usize> = (0..header.tensors.len())
        .filter(|&i| !keep.other().owns(&header.tensors[i].name))
        .collect();
    if kept.len() == header.tensors.len() {
        return Ok(None);
    }
    Ok(Some(Plan {
        header,
        kept,
        regions,
    }))
}

/// Where the copy of `source` that keeps `keep` lives: beside it, so it goes with the model.
pub(super) fn copy_path(source: &Path, keep: Tower) -> PathBuf {
    let stem = source
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    source.with_file_name(format!("{stem}-{}.gguf", keep.name()))
}

/// The bytes a load of `keep` reads from `source`: its copy's tensors, or the whole file when
/// there is nothing to leave out.
pub(super) fn tower_bytes(source: &Path, keep: Tower) -> std::io::Result<u64> {
    Ok(match plan(source, keep)? {
        Some(plan) => plan.data_len(),
        None => std::fs::metadata(source)?.len(),
    })
}

pub(super) struct TowerCopy {
    pub path: PathBuf,
    /// Written by this call, not found current on disk.
    pub written: bool,
}

/// The copy of `source` holding only `keep`, written now unless a current one is on disk.
/// `None` when `source` has no other tower to leave out.
pub(super) fn ensure_copy(source: &Path, keep: Tower) -> std::io::Result<Option<TowerCopy>> {
    let Some(plan) = plan(source, keep)? else {
        return Ok(None);
    };
    let path = copy_path(source, keep);
    let expected_len = plan.header_len(keep) + plan.data_len();
    if is_current(source, &path, expected_len) {
        return Ok(Some(TowerCopy {
            path,
            written: false,
        }));
    }
    let partial = path.with_extension("gguf.partial");
    let written = write_copy(source, &partial, &plan, keep).and_then(|()| {
        let len = std::fs::metadata(&partial)?.len();
        if len != expected_len {
            return Err(invalid("the encoder copy came out the wrong size"));
        }
        std::fs::rename(&partial, &path)
    });
    if let Err(e) = written {
        let _ = std::fs::remove_file(&partial);
        return Err(e);
    }
    Ok(Some(TowerCopy {
        path,
        written: true,
    }))
}

fn is_current(source: &Path, target: &Path, expected_len: u64) -> bool {
    let (Ok(src), Ok(dst)) = (std::fs::metadata(source), std::fs::metadata(target)) else {
        return false;
    };
    let newer = match (src.modified(), dst.modified()) {
        (Ok(s), Ok(d)) => d >= s,
        _ => false,
    };
    newer && dst.len() == expected_len
}

fn write_copy(source: &Path, partial: &Path, plan: &Plan, keep: Tower) -> std::io::Result<()> {
    let header = &plan.header;
    let mut src = File::open(source)?;
    let mut head = vec![0u8; header.end as usize];
    src.read_exact(&mut head)?;
    let slice = |(start, end): (u64, u64)| &head[start as usize..end as usize];

    let mut out = BufWriter::with_capacity(1 << 20, File::create(partial)?);
    out.write_all(&head[..8])?;
    out.write_all(&(plan.kept.len() as u64).to_le_bytes())?;
    out.write_all(&(header.kv.len() as u64).to_le_bytes())?;
    let other = keep.other().flag();
    for entry in &header.kv {
        if entry.key == other {
            out.write_all(&(entry.key.len() as u64).to_le_bytes())?;
            out.write_all(entry.key.as_bytes())?;
            out.write_all(&GGUF_TYPE_BOOL.to_le_bytes())?;
            out.write_all(&[0])?;
        } else {
            out.write_all(slice(entry.span))?;
        }
    }
    let mut offset = 0u64;
    for &i in &plan.kept {
        out.write_all(slice(header.tensors[i].span))?;
        out.write_all(&offset.to_le_bytes())?;
        offset += plan.regions[i];
    }
    let padding = plan.header_len(keep) - plan.unpadded_header_len(keep);
    out.write_all(&vec![0u8; padding as usize])?;

    for &i in &plan.kept {
        src.seek(SeekFrom::Start(
            header.data_start + header.tensors[i].offset,
        ))?;
        let region = plan.regions[i];
        if std::io::copy(&mut (&mut src).take(region), &mut out)? != region {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
    }
    out.into_inner().map_err(|e| e.into_error())?.sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn string(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }

    fn kv_bool(out: &mut Vec<u8>, key: &str, value: bool) {
        string(out, key);
        out.extend(GGUF_TYPE_BOOL.to_le_bytes());
        out.push(u8::from(value));
    }

    /// A two-tower projector: `(name, bytes)` per tensor, each region padded to 32.
    fn projector(dir: &Path, vision: bool, tensors: &[(&str, usize)]) -> PathBuf {
        let mut out = Vec::new();
        out.extend(GGUF_MAGIC);
        out.extend(3u32.to_le_bytes());
        out.extend((tensors.len() as u64).to_le_bytes());
        out.extend(4u64.to_le_bytes());
        string(&mut out, "general.architecture");
        out.extend(GGUF_TYPE_STRING.to_le_bytes());
        string(&mut out, "clip");
        kv_bool(&mut out, "clip.has_vision_encoder", vision);
        kv_bool(&mut out, "clip.has_audio_encoder", true);
        string(&mut out, "clip.vision.image_mean");
        out.extend(GGUF_TYPE_ARRAY.to_le_bytes());
        out.extend(6u32.to_le_bytes());
        out.extend(3u64.to_le_bytes());
        out.extend([0u8; 12]);
        let mut offset = 0u64;
        for (name, len) in tensors {
            string(&mut out, name);
            out.extend(1u32.to_le_bytes());
            out.extend((*len as u64).to_le_bytes());
            out.extend(24u32.to_le_bytes());
            out.extend(offset.to_le_bytes());
            offset = align_up(offset + *len as u64, 32);
        }
        out.resize(align_up(out.len() as u64, 32) as usize, 0);
        for (k, (_, len)) in tensors.iter().enumerate() {
            let start = out.len();
            out.extend(std::iter::repeat_n(k as u8 + 1, *len));
            out.resize(start + align_up(*len as u64, 32) as usize, 0);
        }
        let path = dir.join("mmproj-BF16.gguf");
        std::fs::write(&path, out).unwrap();
        path
    }

    fn tensor_bytes(path: &Path, name: &str) -> Vec<u8> {
        let header = read_header(path).unwrap();
        let i = header.tensors.iter().position(|t| t.name == name).unwrap();
        let start = header.data_start + header.tensors[i].offset;
        let end = header
            .tensors
            .get(i + 1)
            .map_or(header.file_len, |t| header.data_start + t.offset);
        std::fs::read(path).unwrap()[start as usize..end as usize].to_vec()
    }

    fn flag(path: &Path, key: &str) -> Option<bool> {
        read_header(path)
            .unwrap()
            .kv
            .into_iter()
            .find(|e| e.key == key)
            .and_then(|e| e.flag)
    }

    const TENSORS: &[(&str, usize)] = &[
        ("a.blk.0.w", 70),
        ("mm.a.input_projection.weight", 16),
        ("mm.input_projection.weight", 40),
        ("v.blk.0.w", 64),
        ("v.patch_embd.weight", 33),
    ];

    #[test]
    fn an_audio_copy_leaves_out_the_picture_tower_and_says_so() {
        let dir = tempfile::tempdir().unwrap();
        let source = projector(dir.path(), true, TENSORS);
        let copy = ensure_copy(&source, Tower::Audio).unwrap().unwrap();
        assert!(copy.written);
        let copy = copy.path;
        assert_eq!(copy, dir.path().join("mmproj-BF16-audio.gguf"));
        let names: Vec<String> = read_header(&copy)
            .unwrap()
            .tensors
            .into_iter()
            .map(|t| t.name)
            .collect();
        assert_eq!(
            names,
            [
                "a.blk.0.w",
                "mm.a.input_projection.weight",
                "mm.input_projection.weight"
            ]
        );
        assert_eq!(flag(&copy, "clip.has_vision_encoder"), Some(false));
        assert_eq!(flag(&copy, "clip.has_audio_encoder"), Some(true));
        for name in &names {
            assert_eq!(
                tensor_bytes(&copy, name),
                tensor_bytes(&source, name),
                "{name}"
            );
        }
        assert_eq!(
            tower_bytes(&source, Tower::Audio).unwrap(),
            96 + 32 + 64,
            "the bytes a load reads are the kept regions"
        );
    }

    #[test]
    fn a_vision_copy_leaves_out_the_audio_tower_and_its_projector() {
        let dir = tempfile::tempdir().unwrap();
        let source = projector(dir.path(), true, TENSORS);
        let copy = ensure_copy(&source, Tower::Vision).unwrap().unwrap().path;
        let names: Vec<String> = read_header(&copy)
            .unwrap()
            .tensors
            .into_iter()
            .map(|t| t.name)
            .collect();
        assert_eq!(
            names,
            [
                "mm.input_projection.weight",
                "v.blk.0.w",
                "v.patch_embd.weight"
            ]
        );
        assert_eq!(flag(&copy, "clip.has_audio_encoder"), Some(false));
        assert_eq!(
            tensor_bytes(&copy, "v.patch_embd.weight"),
            tensor_bytes(&source, "v.patch_embd.weight")
        );
    }

    #[test]
    fn a_current_copy_is_kept_and_a_stale_one_rewritten() {
        let dir = tempfile::tempdir().unwrap();
        let source = projector(dir.path(), true, TENSORS);
        let copy = ensure_copy(&source, Tower::Audio).unwrap().unwrap().path;
        let first = std::fs::metadata(&copy).unwrap().modified().unwrap();
        let again = ensure_copy(&source, Tower::Audio).unwrap().unwrap();
        assert_eq!(
            (again.path.as_path(), again.written),
            (copy.as_path(), false)
        );
        assert_eq!(std::fs::metadata(&copy).unwrap().modified().unwrap(), first);

        std::fs::write(&copy, b"truncated").unwrap();
        assert!(ensure_copy(&source, Tower::Audio).unwrap().unwrap().written);
        assert_eq!(flag(&copy, "clip.has_vision_encoder"), Some(false));
    }

    #[test]
    fn a_one_tower_projector_needs_no_copy() {
        let dir = tempfile::tempdir().unwrap();
        let source = projector(dir.path(), false, &[("a.blk.0.w", 70)]);
        assert!(ensure_copy(&source, Tower::Audio).unwrap().is_none());
        assert_eq!(
            tower_bytes(&source, Tower::Audio).unwrap(),
            std::fs::metadata(&source).unwrap().len()
        );
        assert!(!copy_path(&source, Tower::Audio).exists());
    }
}
