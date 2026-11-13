//! A prompt with pictures or recordings, laid out one entry a cache position. Media fills its
//! positions with embeddings, which have no token id, so each of those positions gets a stand-in
//! id made from the media's bytes. Equal ids are then equal content at equal positions, and the
//! cache's ledger, its checkpoints and its reuse treat such a prompt like any other.

use llama_cpp_2::token::LlamaToken;

/// One picture or recording in a prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MediaRun {
    /// First position it fills.
    pub start: usize,
    pub len: usize,
    /// Its chunk in the prompt's mtmd chunk list.
    pub chunk: usize,
    /// Hash of the bytes it was made from and its place among the chunks made from them.
    pub content: u64,
}

impl MediaRun {
    pub fn end(&self) -> usize {
        self.start + self.len
    }
}

/// FNV-1a over a media file's bytes and whether it is sound.
pub(super) fn content_hash(bytes: &[u8], audio: bool) -> u64 {
    bytes
        .iter()
        .chain(std::iter::once(&u8::from(audio)))
        .fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
        })
}

/// `content` for the `ordinal`th chunk made from one media file: a long recording is several.
pub(super) fn chunk_content(content: u64, ordinal: usize) -> u64 {
    content ^ (ordinal as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f)
}

/// The id standing in for position `offset` of media `content`. Below -1, so it is never a
/// vocabulary token nor llama.cpp's null token.
pub(super) fn stand_in(content: u64, offset: usize) -> LlamaToken {
    let mixed = (content ^ (offset as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .wrapping_mul(0xff51_afd7_ed55_8ccd);
    LlamaToken(-2 - (mixed >> 34) as i32)
}

pub(super) fn is_stand_in(token: LlamaToken) -> bool {
    token.0 < -1
}

/// Positions ahead of the first picture or recording: all a snapshot may hold, since a snapshot
/// is shared by every prompt that opens the same way.
pub(super) fn text_prefix_len(prompt: &[LlamaToken]) -> usize {
    prompt
        .iter()
        .position(|token| is_stand_in(*token))
        .unwrap_or(prompt.len())
}

/// How far a cache holding `cached` may serve `prompt`: their shared leading entries, stopped at
/// the first picture or recording the two do not hold alike. Stand-in ids are 30 bits, so the
/// media's own hash decides.
pub(super) fn agreed_prefix(
    cached: &[LlamaToken],
    cached_runs: &[MediaRun],
    prompt: &[LlamaToken],
    prompt_runs: &[MediaRun],
) -> usize {
    let shared = cached
        .iter()
        .zip(prompt)
        .take_while(|(cached, prompt)| cached == prompt)
        .count();
    prompt_runs
        .iter()
        .take_while(|run| run.start < shared)
        .find(|run| {
            run.end() > shared
                || !cached_runs
                    .iter()
                    .any(|c| (c.start, c.len, c.content) == (run.start, run.len, run.content))
        })
        .map_or(shared, |run| run.start)
}

/// A stretch of a prompt to decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Span {
    Text {
        from: usize,
        to: usize,
    },
    /// One whole picture or recording, by its index among the prompt's runs.
    Media {
        run: usize,
    },
}

/// `[from, to)` of a prompt as spans in order. `None` when the range cuts a picture or
/// recording, which mtmd only evaluates whole.
pub(super) fn spans(from: usize, to: usize, runs: &[MediaRun]) -> Option<Vec<Span>> {
    let mut out = Vec::new();
    let mut pos = from;
    for (index, run) in runs.iter().enumerate() {
        if run.end() <= from || run.start >= to {
            continue;
        }
        if run.start < pos || run.end() > to {
            return None;
        }
        if run.start > pos {
            out.push(Span::Text {
                from: pos,
                to: run.start,
            });
        }
        out.push(Span::Media { run: index });
        pos = run.end();
    }
    if pos < to {
        out.push(Span::Text { from: pos, to });
    }
    Some(out)
}

/// Where a batch from `pos` that would end at `end` ends instead: ahead of a picture or
/// recording it would cut, or behind it when the batch starts on it.
pub(super) fn batch_end(pos: usize, end: usize, runs: &[MediaRun]) -> usize {
    match runs.iter().find(|run| run.start < end && end < run.end()) {
        Some(run) if run.start > pos => run.start,
        Some(run) => run.end(),
        None => end,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(ids: std::ops::Range<i32>) -> Vec<LlamaToken> {
        ids.map(LlamaToken).collect()
    }

    /// `lead` text tokens, then each of `media` as (content, positions) with `between` text
    /// tokens after it.
    fn prompt(
        lead: usize,
        media: &[(u64, usize)],
        between: usize,
    ) -> (Vec<LlamaToken>, Vec<MediaRun>) {
        let mut ledger = text(0..lead as i32);
        let mut runs = Vec::new();
        for (chunk, (content, len)) in media.iter().enumerate() {
            let start = ledger.len();
            ledger.extend((0..*len).map(|i| stand_in(*content, i)));
            runs.push(MediaRun {
                start,
                len: *len,
                chunk: chunk * 2 + 1,
                content: *content,
            });
            ledger.extend(text(1000..1000 + between as i32));
        }
        (ledger, runs)
    }

    #[test]
    fn a_stand_in_is_never_a_vocabulary_token_and_differs_by_media_and_by_position() {
        for offset in [0usize, 1, 255, 100_000] {
            for content in [0u64, 1, u64::MAX, content_hash(b"a", false)] {
                assert!(is_stand_in(stand_in(content, offset)));
            }
        }
        assert_ne!(stand_in(7, 0), stand_in(7, 1));
        assert_ne!(stand_in(7, 0), stand_in(8, 0));
        assert_ne!(content_hash(b"clip", true), content_hash(b"clip", false));
        assert_ne!(chunk_content(7, 0), chunk_content(7, 1));
        assert!(!is_stand_in(LlamaToken(-1)) && !is_stand_in(LlamaToken(0)));
    }

    #[test]
    fn a_snapshot_stops_at_the_first_picture() {
        let (ledger, _) = prompt(40, &[(1, 8)], 5);
        assert_eq!(text_prefix_len(&ledger), 40);
        assert_eq!(text_prefix_len(&text(0..12)), 12);
    }

    #[test]
    fn a_turn_that_adds_a_picture_is_served_up_to_where_it_differs() {
        let (cached, cached_runs) = prompt(40, &[(1, 8)], 5);
        // The same picture, more text, then a second picture.
        let (mut next, mut next_runs) = prompt(40, &[(1, 8)], 5);
        let more = next.len();
        next.extend(text(2000..2010));
        let start = next.len();
        next.extend((0..6).map(|i| stand_in(2, i)));
        next_runs.push(MediaRun {
            start,
            len: 6,
            chunk: 3,
            content: 2,
        });
        assert_eq!(
            agreed_prefix(&cached, &cached_runs, &next, &next_runs),
            more
        );
    }

    #[test]
    fn a_different_picture_in_the_same_place_ends_the_agreement_where_it_starts() {
        let (cached, cached_runs) = prompt(40, &[(1, 8)], 5);
        let (other, other_runs) = prompt(40, &[(2, 8)], 5);
        assert_eq!(
            agreed_prefix(&cached, &cached_runs, &other, &other_runs),
            40
        );

        // Stand-ins that collide are still two pictures: the hash decides.
        let mut lookalike_runs = cached_runs.clone();
        lookalike_runs[0].content = 99;
        assert_eq!(
            agreed_prefix(&cached, &cached_runs, &cached, &lookalike_runs),
            40
        );
    }

    #[test]
    fn a_text_prompt_and_a_media_cache_agree_up_to_the_media() {
        let (cached, cached_runs) = prompt(40, &[(1, 8)], 5);
        let mut plain = text(0..40);
        plain.extend(text(3000..3020));
        assert_eq!(agreed_prefix(&cached, &cached_runs, &plain, &[]), 40);
        assert_eq!(agreed_prefix(&plain, &[], &cached, &cached_runs), 40);
    }

    #[test]
    fn a_range_is_text_around_whole_pictures_and_never_part_of_one() {
        let (_, runs) = prompt(40, &[(1, 8), (2, 6)], 5);
        // 0..40 text, 40..48 media, 48..53 text, 53..59 media, 59..64 text.
        assert_eq!(
            spans(0, 64, &runs),
            Some(vec![
                Span::Text { from: 0, to: 40 },
                Span::Media { run: 0 },
                Span::Text { from: 48, to: 53 },
                Span::Media { run: 1 },
                Span::Text { from: 59, to: 64 },
            ])
        );
        assert_eq!(
            spans(48, 59, &runs),
            Some(vec![
                Span::Text { from: 48, to: 53 },
                Span::Media { run: 1 }
            ])
        );
        assert_eq!(
            spans(10, 30, &runs),
            Some(vec![Span::Text { from: 10, to: 30 }])
        );
        assert_eq!(spans(44, 64, &runs), None);
        assert_eq!(spans(0, 44, &runs), None);
        assert_eq!(spans(64, 64, &runs), Some(vec![]));
    }

    #[test]
    fn a_batch_ends_ahead_of_a_picture_it_would_cut_or_behind_one_it_starts_on() {
        let (_, runs) = prompt(40, &[(1, 8)], 5);
        assert_eq!(batch_end(0, 44, &runs), 40);
        assert_eq!(batch_end(40, 44, &runs), 48);
        assert_eq!(batch_end(0, 40, &runs), 40);
        assert_eq!(batch_end(0, 53, &runs), 53);
        assert_eq!(batch_end(0, 30, &[]), 30);
    }
}
