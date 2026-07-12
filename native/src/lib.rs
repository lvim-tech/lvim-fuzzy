// lvim-fuzzy-native: OUR OWN fuzzy matcher, from scratch.
//
// Ranking model — an affine-gap local-alignment scorer (fzy/Gotoh class: the
// public "optimal subsequence alignment with match bonuses" recurrence; our own
// integer implementation, no code copied from fzy/frizbee/blink). All completion
// signals are folded INTO the score: first-char / word-boundary / camelCase /
// consecutive-run bonuses and affine leading/inner/trailing gap penalties. The
// final order is a FULLY DETERMINISTIC total order (score desc → shorter
// candidate → source index asc) — no arbitrary equal-score tiebreaks.
//
// Loaded in-process from Lua via LuaJIT FFI (`ffi.load`) — one coarse FFI call
// per keystroke over the whole candidate set. The Lua side (lua/lvim-fuzzy/)
// ships a byte-identical pure-Lua fallback for when this .so is absent.
//
// Performance techniques (measured wins; benchmarked against blink.cmp's frizbee
// matcher — beats it on all tested queries, see README):
//   1. Prepared context (lvimfuzzy_prepare): the candidate set is fixed per
//      completion, so a per-candidate 32-bit byte SIGNATURE + length is built
//      ONCE; per keystroke the prefilter is an O(1) signature-subset test —
//      candidates missing a needle byte are skipped with NO byte scan.
//   2. m == 1 closed form (no DP) + a scalar subsequence prefilter for m ≥ 2.
//   3. Column-major single-pass affine-gap DP with O(m) state on STACK arrays,
//      ping-ponged by reference swap (no per-byte copy, no heap).
//   4. INTER-CANDIDATE AVX2 SIMD (score_tile_avx2): survivors are grouped by
//      length (counting sort) and scored 16-at-a-time — one candidate per i16
//      lane of a 256-bit register — running the SAME affine-gap recurrence in
//      lockstep over a column-major (SoA) tile. Runtime-detected; a scalar tail
//      handles the ragged < 16 remainder and non-AVX2 CPUs. This is what beats
//      frizbee on the broad m ≥ 2 queries.
//   5. Stable O(n) LSD RADIX SORT on a packed (score,len) key — replaces the
//      O(n log n) comparison sort that dominated broad, high-survivor queries.
//   6. Reads survivor bytes straight from the Lua-owned blob (no Vec<&str>
//      rebuild); all buffers reused thread-locally (ZERO per-keystroke heap
//      alloc after warm-up); one coarse FFI call per keystroke.
// The SIMD path is BYTE-IDENTICAL to the scalar path (same scores, same
// deterministic order) — asserted by the `simd_matches_scalar` test over the
// whole query set at 5k and 20k. Same C ABI as the other spikes.

// index-based loops are intentional here — several walk parallel arrays (hits +
// keys, blob + offsets) or need the byte position for bonus/gap math.
#![allow(clippy::needless_range_loop)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::ptr;
use std::slice;

// ─── scoring constants (our own) ────────────────────────────────────────────
// Because max_typos = 0 every survivor matches all m needle chars, so a constant
// per-match reward would not affect RANKING — only bonuses and gaps do. We omit
// it: score = Σ positional bonuses + consecutive bonuses − gap penalties.
const BONUS_FIRST: i32 = 18; // match on candidate byte 0
const BONUS_BOUNDARY: i32 = 16; // match right after / _ - . space :
const BONUS_CAMEL: i32 = 14; // lower→Upper hump
const BONUS_CONSEC: i32 = 16; // match immediately after the previous match
const GAP_LEADING: i32 = -1; // per byte skipped before the first match
const GAP_INNER: i32 = -2; // per byte skipped between matches
const GAP_TRAILING: i32 = -1; // per byte after the last match
const NEG: i32 = i32::MIN / 2; // −∞ that survives one addition

#[inline(always)]
fn lower(b: u8) -> u8 {
    if b.is_ascii_uppercase() { b | 0x20 } else { b }
}

#[inline(always)]
fn is_boundary(b: u8) -> bool {
    matches!(b, b'/' | b'_' | b'-' | b'.' | b' ' | b':')
}

// ─── reused per-call buffers (zero-alloc steady state) ───────────────────────
thread_local! {
    static BUFS: RefCell<Bufs> = const { RefCell::new(Bufs::new()) };
}

struct Bufs {
    hits: Vec<Hit>,
    hits2: Vec<Hit>, // radix ping-pong scratch
    keys: Vec<u32>,  // packed sort key per hit
    lc_needle: Vec<u8>,
    // SIMD-path scratch (all reused across keystrokes; zero steady-state alloc):
    survivors: Vec<u32>,   // signature-passing candidate indices, in index order
    surv_sorted: Vec<u32>, // survivors grouped by candidate length (counting sort)
    count: Vec<u32>,       // per-length counts for the counting sort
    tile: Vec<u8>,         // one SIMD tile, column-major: tile[j*16 + lane]
}

/// (candidate index, score, length) — pushed in index order so a STABLE sort
/// leaves index as the implicit final tiebreak.
type Hit = (u32, i32, u32);

impl Bufs {
    const fn new() -> Self {
        Bufs {
            hits: Vec::new(),
            hits2: Vec::new(),
            keys: Vec::new(),
            lc_needle: Vec::new(),
            survivors: Vec::new(),
            surv_sorted: Vec::new(),
            count: Vec::new(),
            tile: Vec::new(),
        }
    }
}

// ─── prefilter: is `lc_needle` a subsequence of `c` (case-folded)? ───────────
// Scalar, branch-light, early-exit. Candidates are short (identifier length) so
// this stays in L1. In the prepared path it runs only for signature-survivors
// (candidates already known to contain every needle byte).
#[inline]
fn is_subseq(c: &[u8], lc_needle: &[u8]) -> bool {
    let mut ni = 0;
    let nlen = lc_needle.len();
    if nlen == 0 {
        return true;
    }
    let mut want = lc_needle[0];
    for &b in c {
        if lower(b) == want {
            ni += 1;
            if ni == nlen {
                return true;
            }
            want = lc_needle[ni];
        }
    }
    false
}

// ─── m == 1 closed form ──────────────────────────────────────────────────────
// With a single needle char and GAP_LEADING == GAP_TRAILING, the affine-gap
// optimum reduces to: best positional bonus of any matching byte, minus the
// fixed (L-1) gap toll. One branch-light scan, no DP. Returns NEG if absent.
#[inline]
fn score_m1(c: &[u8], want: u8) -> i32 {
    let l = c.len();
    let toll = l as i32 - 1;
    // Byte 0 match is the global max bonus (FIRST) — decide immediately.
    if lower(c[0]) == want {
        return BONUS_FIRST - toll;
    }
    // BONUS_FIRST is now impossible, so the first BOUNDARY match (16) is the best
    // attainable — early-exit on it. Track camel/plain as the fallback.
    let mut best = NEG;
    let mut prev = c[0];
    for j in 1..l {
        let cb = c[j];
        if lower(cb) == want {
            if is_boundary(prev) {
                return BONUS_BOUNDARY - toll;
            } else if prev.is_ascii_lowercase() && cb.is_ascii_uppercase() {
                if BONUS_CAMEL > best {
                    best = BONUS_CAMEL;
                }
            } else if best < 0 {
                best = 0;
            }
        }
        prev = cb;
    }
    if best == NEG { NEG } else { best - toll }
}

// ─── the affine-gap scorer (fzy/Gotoh-class recurrence, our own impl) ────────
// COLUMN-MAJOR single pass: scan candidate bytes left→right ONCE, carrying O(m)
// DP state (two columns of D and M, indexed by needle row). Returns the optimal
// alignment score, or NEG if `lc_needle` is not a subsequence of `c` (so this
// doubles as the match test when no prefilter ran).
//   D[i] = best score with needle[i] matched at the current column.
//   M[i] = best score for needle[0..=i] using candidate[0..=current column].
const MAXM: usize = 64; // fast-path needle length; longer needles use the Vec fallback

#[inline]
fn score(c: &[u8], lc_needle: &[u8]) -> i32 {
    let m = lc_needle.len();
    if m > MAXM {
        return score_big(c, lc_needle);
    }
    // Two columns of D and M on the stack; ping-pong by swapping REFERENCES each
    // byte (pointer swap, no array copy). O(m) hot state, L1-resident.
    let (mut d0, mut d1) = ([NEG; MAXM], [NEG; MAXM]);
    let (mut m0, mut m1) = ([NEG; MAXM], [NEG; MAXM]);
    let mut dp: &mut [i32; MAXM] = &mut d0; // previous column (init: unreachable)
    let mut mp: &mut [i32; MAXM] = &mut m0;
    let mut dc: &mut [i32; MAXM] = &mut d1; // current column
    let mut mc: &mut [i32; MAXM] = &mut m1;
    let last = m - 1;
    for (j, &cb) in c.iter().enumerate() {
        let lc = lower(cb);
        let bonus_j = if j == 0 {
            BONUS_FIRST
        } else {
            let prev = c[j - 1];
            if is_boundary(prev) {
                BONUS_BOUNDARY
            } else if prev.is_ascii_lowercase() && cb.is_ascii_uppercase() {
                BONUS_CAMEL
            } else {
                0
            }
        };
        for i in 0..m {
            let dval = if lc == lc_needle[i] {
                let fresh = if i == 0 {
                    (j as i32) * GAP_LEADING
                } else {
                    mp[i - 1]
                };
                let consec = if i == 0 { NEG } else { dp[i - 1] };
                (fresh + bonus_j).max(consec + bonus_j + BONUS_CONSEC)
            } else {
                NEG
            };
            let gap = if i == last { GAP_TRAILING } else { GAP_INNER };
            dc[i] = dval;
            mc[i] = dval.max(mp[i] + gap);
        }
        std::mem::swap(&mut dp, &mut dc);
        std::mem::swap(&mut mp, &mut mc);
    }
    mp[last] // after the final swap the last column lives in the "prev" slot
}

/// Vec-backed fallback for needles longer than MAXM (rare in completion). Local
/// allocation is fine here — this path is essentially never hit for real needles.
#[inline(never)]
fn score_big(c: &[u8], lc_needle: &[u8]) -> i32 {
    let m = lc_needle.len();
    let mut m_prev = vec![NEG; m];
    let mut m_cur = vec![NEG; m];
    let mut d_prev = vec![NEG; m];
    let mut d_cur = vec![NEG; m];
    let last = m - 1;
    for (j, &cb) in c.iter().enumerate() {
        let lc = lower(cb);
        let bonus_j = if j == 0 {
            BONUS_FIRST
        } else if is_boundary(c[j - 1]) {
            BONUS_BOUNDARY
        } else if c[j - 1].is_ascii_lowercase() && cb.is_ascii_uppercase() {
            BONUS_CAMEL
        } else {
            0
        };
        let mut carry_d = NEG;
        let mut carry_m = (j as i32) * GAP_LEADING;
        for i in 0..m {
            let dval = if lc == lc_needle[i] {
                (carry_m + bonus_j).max(carry_d + bonus_j + BONUS_CONSEC)
            } else {
                NEG
            };
            let gap = if i == last { GAP_TRAILING } else { GAP_INNER };
            carry_d = d_prev[i];
            carry_m = m_prev[i];
            d_cur[i] = dval;
            m_cur[i] = dval.max(m_prev[i] + gap);
        }
        std::mem::swap(&mut m_prev, &mut m_cur);
        std::mem::swap(&mut d_prev, &mut d_cur);
    }
    m_prev[last]
}

// ─── inter-candidate SIMD (AVX2): score 16 candidates per instruction ────────
// The SAME affine-gap recurrence as `score`, but run across 16 EQUAL-LENGTH
// candidates in the 16 i16 lanes of a 256-bit register (one lane per candidate).
// Candidates are laid out column-major in `tile` (tile[j*16 + lane] = byte j of
// candidate `lane`), so column j is a contiguous 16-byte load. bonus_j is a
// per-lane vector (each candidate has its own boundary/camel structure); the
// needle char is a broadcast scalar. Byte-identical to the scalar path: scores
// are small integers, and using i16::MIN as the −∞ sentinel with SATURATING
// signed adds keeps unreachable lanes pinned far below any real score (real
// scores are bounded by ~32·m ≪ i16::MAX), so the max() picks the same real
// alignment the scalar i32 path does. Verified by the exhaustive diff test.
#[cfg(target_arch = "x86_64")]
const SIMD_W: usize = 16;
#[cfg(target_arch = "x86_64")]
const SIMD_M: usize = 16; // SIMD path handles needles up to this length; longer → scalar
// (keeps the per-tile DP-state init small; completion needles are far shorter)

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)] // an all-intrinsics SIMD kernel; the whole body is unsafe by nature
unsafe fn score_tile_avx2(tile: &[u8], l: usize, lc_needle: &[u8], out: &mut [i16; SIMD_W]) {
    use core::arch::x86_64::*;
    let m = lc_needle.len();
    let last = m - 1;
    let neg = _mm256_set1_epi16(i16::MIN);

    // DP state: two columns (prev/cur) of D and M, one i16 lane per candidate,
    // indexed by needle row. Stack arrays, ping-ponged by reference swap.
    let mut d0 = [neg; SIMD_M];
    let mut d1 = [neg; SIMD_M];
    let mut m0 = [neg; SIMD_M];
    let mut m1 = [neg; SIMD_M];
    let mut dp: &mut [__m256i; SIMD_M] = &mut d0;
    let mut mp: &mut [__m256i; SIMD_M] = &mut m0;
    let mut dc: &mut [__m256i; SIMD_M] = &mut d1;
    let mut mc: &mut [__m256i; SIMD_M] = &mut m1;

    let v20 = _mm256_set1_epi16(0x20);
    let v64 = _mm256_set1_epi16(64); // 'A' - 1
    let v91 = _mm256_set1_epi16(91); // 'Z' + 1
    let v96 = _mm256_set1_epi16(96); // 'a' - 1
    let v123 = _mm256_set1_epi16(123); // 'z' + 1
    let vfirst = _mm256_set1_epi16(BONUS_FIRST as i16);
    let vbound = _mm256_set1_epi16(BONUS_BOUNDARY as i16);
    let vcamel = _mm256_set1_epi16(BONUS_CAMEL as i16);
    let vconsec = _mm256_set1_epi16(BONUS_CONSEC as i16);
    let vzero = _mm256_setzero_si256();
    let bnd = [
        _mm256_set1_epi16(47), // /
        _mm256_set1_epi16(95), // _
        _mm256_set1_epi16(45), // -
        _mm256_set1_epi16(46), // .
        _mm256_set1_epi16(32), // space
        _mm256_set1_epi16(58), // :
    ];

    // load column j (16 bytes) zero-extended to 16×i16
    let load_col = |j: usize| -> __m256i {
        let p = tile.as_ptr().add(j * SIMD_W) as *const __m128i;
        _mm256_cvtepu8_epi16(_mm_loadu_si128(p))
    };
    let is_upper = |cb: __m256i| -> __m256i {
        _mm256_and_si256(_mm256_cmpgt_epi16(cb, v64), _mm256_cmpgt_epi16(v91, cb))
    };
    let is_lower = |cb: __m256i| -> __m256i {
        _mm256_and_si256(_mm256_cmpgt_epi16(cb, v96), _mm256_cmpgt_epi16(v123, cb))
    };
    let lower =
        |cb: __m256i| -> __m256i { _mm256_add_epi16(cb, _mm256_and_si256(is_upper(cb), v20)) };

    let mut prev_raw = vzero;
    for j in 0..l {
        let cb = load_col(j);
        let lc = lower(cb);
        let bonus = if j == 0 {
            vfirst
        } else {
            let mut bmask = _mm256_cmpeq_epi16(prev_raw, bnd[0]);
            for vb in &bnd[1..] {
                bmask = _mm256_or_si256(bmask, _mm256_cmpeq_epi16(prev_raw, *vb));
            }
            let camel = _mm256_and_si256(is_lower(prev_raw), is_upper(cb));
            // precedence FIRST > BOUNDARY > CAMEL > 0 (boundary applied last → wins)
            let t = _mm256_blendv_epi8(vzero, vcamel, camel);
            _mm256_blendv_epi8(t, vbound, bmask)
        };
        let vlead = _mm256_set1_epi16(-(j as i16)); // j * GAP_LEADING (= -1)
        for i in 0..m {
            let want = _mm256_set1_epi16(lc_needle[i] as i16);
            let matched = _mm256_cmpeq_epi16(lc, want);
            let fresh = if i == 0 { vlead } else { mp[i - 1] };
            let consec = if i == 0 { neg } else { dp[i - 1] };
            let a = _mm256_adds_epi16(fresh, bonus);
            let b = _mm256_adds_epi16(_mm256_adds_epi16(consec, bonus), vconsec);
            let cand_d = _mm256_max_epi16(a, b);
            let dval = _mm256_blendv_epi8(neg, cand_d, matched);
            let gap = _mm256_set1_epi16(if i == last {
                GAP_TRAILING as i16
            } else {
                GAP_INNER as i16
            });
            dc[i] = dval;
            mc[i] = _mm256_max_epi16(dval, _mm256_adds_epi16(mp[i], gap));
        }
        std::mem::swap(&mut dp, &mut dc);
        std::mem::swap(&mut mp, &mut mc);
        prev_raw = cb;
    }
    // after the final swap, the last column is in `mp`; store lane scores
    _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, mp[last]);
}

// Scores above this (as i32) are real matches; the SIMD i16 sentinel/unreachable
// states sit far below (near i16::MIN), the scalar ones below NEG/2 — either way
// cleanly separated from real scores (bounded to ~±(32·m + 2·L), ≪ 16000).
const MATCH_FLOOR: i32 = -16000;

/// Score one length-`l` RUN of survivor candidates: AVX2 tiles of 16 where
/// available, scalar tail for the remainder (and the whole run without AVX2).
/// Pushes matched `(idx, score, len)` hits. Byte-identical scores to the pure
/// scalar path (verified by the diff test).
#[allow(unused_variables, clippy::too_many_arguments)]
fn score_run(
    run: &[u32],
    l: usize,
    blob: &[u8],
    offs: &[i32],
    lc_needle: &[u8],
    use_simd: bool,
    tile: &mut Vec<u8>,
    hits: &mut Vec<Hit>,
) {
    let m = lc_needle.len();
    let rn = run.len();
    if rn == 0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if use_simd && m <= SIMD_M && l > 0 && is_x86_feature_detected!("avx2") {
        if tile.len() < l * SIMD_W {
            tile.resize(l * SIMD_W, 0);
        }
        // Every bucket is scored by full 16-wide tiles; a partial LAST tile is
        // PADDED by repeating the bucket's last real candidate (a valid same-
        // length load) and its output lanes are ignored — no scalar tail waste.
        let mut i = 0usize;
        while i < rn {
            let this = (rn - i).min(SIMD_W); // real candidates in this tile
            for lane in 0..SIMD_W {
                let src = run[i + if lane < this { lane } else { this - 1 }] as usize;
                let s = offs[src] as usize;
                let cand = &blob[s..s + l];
                for j in 0..l {
                    tile[j * SIMD_W + lane] = cand[j];
                }
            }
            let mut out = [0i16; SIMD_W];
            // SAFETY: avx2 is feature-detected above; tile holds l*16 bytes.
            unsafe { score_tile_avx2(&tile[..l * SIMD_W], l, lc_needle, &mut out) };
            for lane in 0..this {
                let sc = out[lane] as i32;
                if sc > MATCH_FLOOR {
                    hits.push((run[i + lane], sc, l as u32));
                }
            }
            i += this;
        }
        return;
    }
    // scalar path (non-AVX2 CPUs / needles longer than the SIMD width)
    for &ri in run {
        let s = offs[ri as usize] as usize;
        let sc = score(&blob[s..s + l], lc_needle);
        if sc > MATCH_FLOOR {
            hits.push((ri, sc, l as u32));
        }
    }
}

// ─── deterministic O(n) ordering: stable LSD radix sort ──────────────────────
// Ranking order is a TOTAL, deterministic order: score desc → shorter candidate
// → index asc. We pack (score, len) into one u32 ascending key (higher score →
// lower key, shorter len → lower key); a stable radix keeps equal keys in push
// (index) order, giving the index tiebreak for free. O(n) beats the comparison
// sort's O(n log n) on broad queries with thousands of survivors.
const SCORE_BIAS: i32 = 1 << 18; // maps our bounded scores to a non-negative range

#[inline]
fn pack_key(score: i32, len: u32) -> u32 {
    // (BIAS - score) clamped to 20 bits, then 12 bits of length (clamped 0xFFF):
    // higher score → lower key → ranked first; shorter → lower key. The score
    // field sits in bits 12..32, so the radix below MUST cover all 4 bytes.
    let s = (SCORE_BIAS - score).clamp(0, 0xFFFFF) as u32;
    (s << 12) | len.min(0xFFF)
}

/// Stable ascending LSD radix sort of `hits` by `pack_key`, writing the top
/// `cap` results to the output slices. Uses reused ping-pong + counting buffers.
#[inline]
fn radix_finalize(b: &mut Bufs, out_i: &mut [i32], out_s: &mut [i32]) -> i32 {
    let n = b.hits.len();
    let cap = out_i.len();
    if n == 0 {
        return 0;
    }
    // cache the packed keys once (parallel to hits), then move keys alongside the
    // payload each pass so pack_key is computed only n times, not 3n.
    b.keys.clear();
    b.keys.reserve(n);
    for &(_, sc, len) in &b.hits {
        b.keys.push(pack_key(sc, len));
    }
    if b.hits2.len() < n {
        b.hits2.resize(n, (0, 0, 0));
    }
    // 4 LSD passes of 8 bits cover the full 32-bit key (12-bit length + up-to-20-bit
    // score field). Each pass writes into hits2 then swaps, so the sorted payload
    // always ends back in b.hits.
    for &shift in &[0u32, 8, 16, 24] {
        let mut count = [0u32; 256];
        for &key in b.keys.iter().take(n) {
            count[((key >> shift) & 0xFF) as usize] += 1;
        }
        let mut acc = 0u32;
        for c in count.iter_mut() {
            let t = *c;
            *c = acc;
            acc += t;
        }
        for i in 0..n {
            let k = ((b.keys[i] >> shift) & 0xFF) as usize;
            let pos = count[k] as usize;
            count[k] += 1;
            b.hits2[pos] = b.hits[i];
        }
        std::mem::swap(&mut b.hits, &mut b.hits2);
        // reorder keys to match the payload's new order for the next pass
        if shift != 24 {
            for i in 0..n {
                b.keys[i] = pack_key(b.hits[i].1, b.hits[i].2);
            }
        }
    }
    let cnt = n.min(cap);
    for k in 0..cnt {
        let (idx, sc, _) = b.hits[k];
        out_i[k] = idx as i32;
        out_s[k] = sc;
    }
    cnt as i32
}

/// Fold per-candidate score offsets into the hits BEFORE ranking (ABI 6). `boosts[idx]` is
/// added to candidate `idx`'s match score, letting a caller (lvim-cmp) bias the fuzzy order
/// with signals that live outside the string match — exact-prefix, proximity, frecency —
/// while still respecting the fuzzy score. Indices past the slice stay unboosted; the add
/// saturates so an extreme boost can never wrap the score.
#[inline]
fn apply_boosts(hits: &mut [Hit], boosts: Option<&[i32]>) {
    if let Some(bo) = boosts {
        for h in hits.iter_mut() {
            let idx = h.0 as usize;
            if idx < bo.len() {
                h.1 = h.1.saturating_add(bo[idx]);
            }
        }
    }
}

// ─── orchestration ───────────────────────────────────────────────────────────
#[inline]
fn run_match(
    blob: &[u8],
    offsets: &[i32],
    n: usize,
    needle: &[u8],
    out_idx: &mut [i32],
    out_score: &mut [i32],
) -> i32 {
    BUFS.with(|cell| {
        let mut b = cell.borrow_mut();
        // case-fold the needle once
        b.lc_needle.clear();
        b.lc_needle.extend(needle.iter().map(|&x| lower(x)));
        b.hits.clear();

        // SAFETY: we read lc_needle immutably while mutating other Bufs fields;
        // split the borrow by taking the needle into a local slice via a raw ptr
        // into the (stable, not-resized-here) lc_needle buffer.
        let (nptr, nlen) = (b.lc_needle.as_ptr(), b.lc_needle.len());
        let lc_needle: &[u8] = unsafe { slice::from_raw_parts(nptr, nlen) };
        if nlen == 0 {
            let cap = out_idx.len().min(n);
            for i in 0..cap {
                out_idx[i] = i as i32;
                out_score[i] = 0;
            }
            return cap as i32;
        }

        const MATCHED: i32 = NEG / 2; // scores above this are real matches
        if nlen == 1 {
            // m == 1: fused prefilter + closed-form score, one scan per candidate.
            let want = lc_needle[0];
            for i in 0..n {
                let s = offsets[i] as usize;
                let e = offsets[i + 1] as usize;
                let sc = score_m1(&blob[s..e], want);
                if sc > MATCHED {
                    b.hits.push((i as u32, sc, (e - s) as u32));
                }
            }
        } else {
            // m ≥ 2: cheap subsequence prefilter rejects the majority, then the
            // column-major affine-gap DP scores each survivor (one scan each).
            for i in 0..n {
                let s = offsets[i] as usize;
                let e = offsets[i + 1] as usize;
                let c = &blob[s..e];
                if c.len() < nlen || !is_subseq(c, lc_needle) {
                    continue;
                }
                let sc = score(c, lc_needle);
                if sc > MATCHED {
                    b.hits.push((i as u32, sc, (e - s) as u32));
                }
            }
        }

        radix_finalize(&mut b, out_idx, out_score)
    })
}

// ─── prepared context (the real-plugin path: candidate set fixed per completion,
//     needle changes per keystroke) ─────────────────────────────────────────────
// Per completion context we build ONCE: a copy of the candidate blob + offsets, a
// per-candidate byte-presence SIGNATURE (a 32-bit bloom: bit = folded lowercased
// byte), and per-candidate length. Per keystroke the prefilter becomes an O(1)
// signature-subset test — no candidate byte scan for the majority that cannot
// match — so only true survivors reach the DP. All storage is reused across
// contexts (grown, never freed). blink+frizbee likewise cache the haystack, so
// this is the fair steady-state comparison.

/// 32-bit presence signature of one byte (case-folded; non-alpha folded to bit 26).
#[inline(always)]
fn sig_byte(b: u8) -> u32 {
    let l = lower(b);
    if l.is_ascii_lowercase() {
        1u32 << (l - b'a')
    } else if l.is_ascii_digit() {
        1u32 << 26
    } else {
        1u32 << (27 + (l as u32 & 3)) // spread the few symbol classes over bits 27..30
    }
}

struct Ctx {
    blob: Vec<u8>,
    offs: Vec<i32>,
    sigs: Vec<u32>,
    lens: Vec<u32>,
    n: usize,
    // Partial trailing line carried between blob-ingest chunks (ABI 5): a producer's stdout arrives in
    // arbitrary byte chunks that may split a line, so the bytes after the last '\n' of a chunk wait here for
    // the next chunk (or the final flush). Unused by the offsets-based prepare/append path.
    pending: Vec<u8>,
}

impl Ctx {
    const fn new() -> Self {
        Ctx {
            blob: Vec::new(),
            offs: Vec::new(),
            sigs: Vec::new(),
            lens: Vec::new(),
            n: 0,
            pending: Vec::new(),
        }
    }
}

thread_local! {
    static CTX: RefCell<Ctx> = const { RefCell::new(Ctx::new()) };
}

/// Cooperative CHUNKED match state (ABI 3): the caller drives the match over `[0, n)` in slices across event
/// loop ticks, so a huge-pool match never blocks the editor in one FFI call. `begin` records the lowercased
/// needle + its signature and clears the hit accumulator; each `step` scans one candidate range (appending to
/// the shared `b.hits`); `finish` radix-sorts the accumulated hits into the caller's output — byte-identical
/// to the single-shot `match_prepared`, because `scan_range` processes candidates in index order in both, so
/// equal `(score, len)` ties finalize in index-ascending order either way. Same thread as `prepare`.
struct ChunkState {
    lc_needle: Vec<u8>,
    nsig: u32,
    cursor: usize,
    n: usize,
}

impl ChunkState {
    const fn new() -> Self {
        ChunkState {
            lc_needle: Vec::new(),
            nsig: 0,
            cursor: 0,
            n: 0,
        }
    }
}

thread_local! {
    static CHUNK: RefCell<ChunkState> = const { RefCell::new(ChunkState::new()) };
}

/// Append `n` candidates (their own blob + 0-based `offsets`, n+1 entries) to `c`
/// IN PLACE — extends the SoA blob/offs/sigs/lens with only the new candidates'
/// data; existing entries are never re-scanned. This is the core shared by both
/// `prepare` (which clears first) and `append`, so the resulting context is
/// byte-identical whether a set was prepared whole or grown incrementally.
fn ctx_append(c: &mut Ctx, blob: &[u8], offsets: &[i32], n: usize) {
    if c.offs.is_empty() {
        c.offs.push(0); // leading 0 = start of the first candidate
    }
    let base = c.blob.len() as i32; // global byte offset the new blob starts at
    c.blob.extend_from_slice(blob);
    c.sigs.reserve(n);
    c.lens.reserve(n);
    c.offs.reserve(n);
    for i in 0..n {
        let s = offsets[i] as usize;
        let e = offsets[i + 1] as usize;
        c.offs.push(base + offsets[i + 1]); // absolute END offset of candidate i
        let mut m = 0u32;
        for &bb in &blob[s..e] {
            m |= sig_byte(bb);
        }
        c.sigs.push(m);
        c.lens.push((e - s) as u32);
    }
    c.n += n;
}

// ─── blob-ingest handle contexts (ABI 5): candidates never become Lua strings ─
// A streaming finder over a huge tree (millions of paths) would otherwise materialise every path as a Lua
// string + table on the main thread — a multi-second load of string-intern-table doublings + GC pauses. The
// blob API removes the Lua-side pool entirely: the producer's raw stdout bytes are handed straight to the
// native context, which SPLITS on '\n' and stores each line as a candidate; the Lua side keeps only the
// visible / top-K rows, materialised on demand via `text`. Contexts here are keyed by an integer HANDLE in a
// registry (not the single active slot the prepare/append path uses), so a picker's huge pool can never be
// clobbered by another consumer preparing in between — and, unlike the single slot, it need not be
// re-uploadable from Lua (the Lua side no longer holds the strings to re-upload).

/// Push ONE candidate (its raw bytes) onto `c`, computing its signature + length. Empty lines are SKIPPED (a
/// fuzzy candidate is never the empty string; this mirrors the finders' own `p ~= ""` guard), so the ingested
/// set is byte-identical to preparing the same non-empty lines. Offsets are absolute END offsets with a
/// leading 0 — identical layout to `ctx_append` / `build_ctx`, so a blob-ingested context matches identically.
fn ctx_push(c: &mut Ctx, bytes: &[u8]) {
    if bytes.is_empty() {
        return;
    }
    if c.offs.is_empty() {
        c.offs.push(0);
    }
    c.blob.extend_from_slice(bytes);
    c.offs.push(c.blob.len() as i32);
    let mut m = 0u32;
    for &bb in bytes {
        m |= sig_byte(bb);
    }
    c.sigs.push(m);
    c.lens.push(bytes.len() as u32);
    c.n += 1;
}

/// Ingest a raw byte chunk into `c`, splitting on '\n'. A line straddling the chunk boundary is carried in
/// `c.pending` until the next chunk (or `ctx_flush_blob`) completes it — so chunking at ANY byte boundary
/// yields the same candidate set. Existing candidates are never re-scanned (incremental / streaming).
fn ctx_append_blob(c: &mut Ctx, data: &[u8]) {
    let mut start = 0;
    for i in 0..data.len() {
        if data[i] == b'\n' {
            if c.pending.is_empty() {
                ctx_push(c, &data[start..i]);
            } else {
                let mut line = std::mem::take(&mut c.pending);
                line.extend_from_slice(&data[start..i]);
                ctx_push(c, &line);
            }
            start = i + 1;
        }
    }
    if start < data.len() {
        c.pending.extend_from_slice(&data[start..]);
    }
}

/// Flush a trailing partial line (a final line with no terminating '\n') as its own candidate.
fn ctx_flush_blob(c: &mut Ctx) {
    if !c.pending.is_empty() {
        let line = std::mem::take(&mut c.pending);
        ctx_push(c, &line);
    }
}

/// Registry of handle-addressed blob contexts, keyed by a monotonically-issued id.
struct Registry {
    map: HashMap<u32, Ctx>,
    next: u32,
}

impl Registry {
    fn new() -> Self {
        Registry {
            map: HashMap::new(),
            next: 1,
        }
    }
}

thread_local! {
    static REGISTRY: RefCell<Registry> = RefCell::new(Registry::new());
}

/// Create a new empty blob context and return its handle (a positive id). (ABI ≥ 5.)
#[unsafe(no_mangle)]
pub extern "C" fn lvimfuzzy_ctx_new() -> i32 {
    REGISTRY.with(|r| {
        let mut reg = r.borrow_mut();
        let id = reg.next;
        reg.next += 1;
        reg.map.insert(id, Ctx::new());
        id as i32
    })
}

/// Ingest `len` raw bytes into handle `id`, splitting on '\n' (a straddling line is carried to the next
/// chunk). Returns the new candidate count. (ABI ≥ 5.)
///
/// # Safety
/// `data` must be valid for `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lvimfuzzy_ctx_append_blob(id: i32, data: *const u8, len: i32) -> i32 {
    if data.is_null() || len <= 0 {
        return REGISTRY.with(|r| r.borrow().map.get(&(id as u32)).map(|c| c.n as i32).unwrap_or(0));
    }
    let bytes = unsafe { slice::from_raw_parts(data, len as usize) };
    REGISTRY.with(|r| {
        let mut reg = r.borrow_mut();
        match reg.map.get_mut(&(id as u32)) {
            Some(c) => {
                ctx_append_blob(c, bytes);
                c.n as i32
            }
            None => 0,
        }
    })
}

/// Flush handle `id`'s trailing partial line (call at end of stream). Returns the candidate count. (ABI ≥ 5.)
#[unsafe(no_mangle)]
pub extern "C" fn lvimfuzzy_ctx_flush(id: i32) -> i32 {
    REGISTRY.with(|r| {
        let mut reg = r.borrow_mut();
        match reg.map.get_mut(&(id as u32)) {
            Some(c) => {
                ctx_flush_blob(c);
                c.n as i32
            }
            None => 0,
        }
    })
}

/// The candidate count of handle `id` (0 if unknown). (ABI ≥ 5.)
#[unsafe(no_mangle)]
pub extern "C" fn lvimfuzzy_ctx_count(id: i32) -> i32 {
    REGISTRY.with(|r| r.borrow().map.get(&(id as u32)).map(|c| c.n as i32).unwrap_or(0))
}

/// Pointer to candidate `idx` (1-based) of handle `id`; its byte length is written to `*out_len`. Returns a
/// pointer INTO the context's blob (no copy) — valid until the next ingest/flush/drop of this handle mutates
/// it, so the caller must copy it out immediately (the Lua side `ffi.string`s it before any further ingest).
/// Null (and `*out_len = 0`) for an unknown handle or out-of-range index. (ABI ≥ 5.)
///
/// # Safety
/// `out_len` must be a valid `*mut i32`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lvimfuzzy_ctx_text(id: i32, idx: i32, out_len: *mut i32) -> *const u8 {
    if out_len.is_null() {
        return ptr::null();
    }
    REGISTRY.with(|r| {
        let reg = r.borrow();
        if let Some(c) = reg.map.get(&(id as u32))
            && idx >= 1
            && (idx as usize) <= c.n
        {
            let i = idx as usize;
            let s = c.offs[i - 1] as usize;
            let e = c.offs[i] as usize;
            unsafe {
                *out_len = (e - s) as i32;
            }
            return unsafe { c.blob.as_ptr().add(s) };
        }
        unsafe {
            *out_len = 0;
        }
        ptr::null()
    })
}

/// PARALLEL prepared match against handle `id` (same ranking + output contract as `lvimfuzzy_match_prepared_par`,
/// but over a handle-addressed context). Returns the number of rows WRITTEN (≤ `max_out`); also writes the TRUE
/// total match count (before the cap) to `*out_total` when non-null — so a consumer can show fzf-style
/// `matched/total` even though only the top `max_out` rows are materialised. Byte-identical results to the
/// single-slot path. (ABI ≥ 5.)
///
/// # Safety
/// `needle` valid for `needle_len`; out arrays hold `max_out` elements; `out_total` null or a valid `*mut i32`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lvimfuzzy_ctx_match_par(
    id: i32,
    needle: *const u8,
    needle_len: i32,
    out_idx: *mut i32,
    out_score: *mut i32,
    max_out: i32,
    out_total: *mut i32,
) -> i32 {
    if needle.is_null() || out_idx.is_null() || out_score.is_null() || max_out <= 0 {
        return 0;
    }
    unsafe {
        let needle = slice::from_raw_parts(needle, needle_len.max(0) as usize);
        let cap = max_out as usize;
        let out_i = slice::from_raw_parts_mut(out_idx, cap);
        let out_s = slice::from_raw_parts_mut(out_score, cap);
        let (written, total) = REGISTRY.with(|r| {
            let reg = r.borrow();
            match reg.map.get(&(id as u32)) {
                Some(c) => BUFS.with(|bc| parallel_prepared_run(c, &mut bc.borrow_mut(), needle, true, out_i, out_s)),
                None => (0, 0),
            }
        });
        if !out_total.is_null() {
            *out_total = total;
        }
        written
    }
}

/// Drop handle `id`, releasing its context memory. (ABI ≥ 5.)
#[unsafe(no_mangle)]
pub extern "C" fn lvimfuzzy_ctx_drop(id: i32) {
    REGISTRY.with(|r| {
        r.borrow_mut().map.remove(&(id as u32));
    });
}

/// Build the per-completion context ONCE (candidate set fixed; needle varies per
/// keystroke): copies the blob + offsets and precomputes a byte signature + length
/// per candidate. Returns n.
///
/// # Safety
/// `hay`/`offs` must be valid for the stated lengths (offsets has n+1 elements,
/// blob length == offsets[n]).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lvimfuzzy_prepare(hay: *const u8, offs: *const i32, n: i32) -> i32 {
    if hay.is_null() || offs.is_null() || n <= 0 {
        return 0;
    }
    unsafe {
        let n = n as usize;
        let offsets = slice::from_raw_parts(offs, n + 1);
        let total = offsets[n] as usize;
        let blob = slice::from_raw_parts(hay, total);
        CTX.with(|cell| {
            let mut c = cell.borrow_mut();
            c.blob.clear();
            c.offs.clear();
            c.sigs.clear();
            c.lens.clear();
            c.n = 0;
            ctx_append(&mut c, blob, offsets, n);
        });
        n as i32
    }
}

/// Append `n` NEW candidates (their own blob + n+1 offsets) to the ACTIVE prepared
/// context, extending it in place — only the new candidates are scanned, the
/// existing entries are untouched. Returns the new total candidate count. A
/// subsequent match() ranks the whole grown set. (ABI ≥ 2.)
///
/// # Safety
/// `hay`/`offs` must be valid for the stated lengths (offsets has n+1 elements,
/// blob length == offsets[n]).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lvimfuzzy_append(hay: *const u8, offs: *const i32, n: i32) -> i32 {
    if hay.is_null() || offs.is_null() || n <= 0 {
        return CTX.with(|cell| cell.borrow().n as i32);
    }
    unsafe {
        let n = n as usize;
        let offsets = slice::from_raw_parts(offs, n + 1);
        let total = offsets[n] as usize;
        let blob = slice::from_raw_parts(hay, total);
        CTX.with(|cell| {
            let mut c = cell.borrow_mut();
            ctx_append(&mut c, blob, offsets, n);
            c.n as i32
        })
    }
}

/// Free the active prepared context's memory (clears + releases the SoA Vecs).
/// Optional — a consumer that has finished with a large pool can reclaim it
/// eagerly instead of waiting for the next `prepare` to overwrite it. (ABI ≥ 2.)
#[unsafe(no_mangle)]
pub extern "C" fn lvimfuzzy_ctx_free() {
    CTX.with(|cell| {
        let mut c = cell.borrow_mut();
        c.blob = Vec::new();
        c.offs = Vec::new();
        c.sigs = Vec::new();
        c.lens = Vec::new();
        c.n = 0;
    });
}

/// Core prepared match (shared by the SIMD and scalar-forced entry points).
/// m == 1 → closed form; m ≥ 2 → signature prefilter → survivors grouped by
/// length (counting sort) → per-length runs scored by `score_run` (SIMD tiles +
/// scalar tail) → O(n) radix sort. `use_simd = false` forces the scalar path (the
/// diff test drives both and asserts byte-identical output).
/// Scan candidates `[start, end)` for a >=1-byte needle (its lowercased bytes `lc_needle` + signature `nsig`):
/// O(1) signature reject, cheap subsequence confirm, length-bucket the survivors, SIMD-score them, and APPEND
/// `(idx, score, len)` to `b.hits`. Does NOT clear hits or finalize — so it composes: one call over `[0, n)`
/// (prepared_run) or many range calls whose accumulated hits finalize identically (the chunked async path).
/// The per-range survivor scratch is reused; hits accumulate. `lc_needle` must outlive the call (a stable
/// view — never `b.lc_needle`, which score_run may not touch anyway).
fn scan_range(c: &Ctx, b: &mut Bufs, lc_needle: &[u8], nsig: u32, use_simd: bool, start: usize, end: usize) {
    let nlen = lc_needle.len();
    if nlen == 1 {
        let want = lc_needle[0];
        for i in start..end {
            if c.sigs[i] & nsig != nsig {
                continue;
            }
            let s = c.offs[i] as usize;
            let e = c.offs[i + 1] as usize;
            let sc = score_m1(&c.blob[s..e], want);
            if sc > MATCH_FLOOR {
                b.hits.push((i as u32, sc, c.lens[i]));
            }
        }
        return;
    }
    // Collect this range's survivors: O(1) signature reject, then a cheap subsequence confirm (rejects
    // wrong-order candidates the DP would only reject after a full score — pruning them shrinks the SIMD tile
    // count for broad queries; for selective queries the survivor set is tiny so this is nearly free).
    b.survivors.clear();
    let mut max_len = 0u32;
    for i in start..end {
        if c.sigs[i] & nsig == nsig {
            let s = c.offs[i] as usize;
            let e = c.offs[i + 1] as usize;
            if is_subseq(&c.blob[s..e], lc_needle) {
                b.survivors.push(i as u32);
                if c.lens[i] > max_len {
                    max_len = c.lens[i];
                }
            }
        }
    }
    let sn = b.survivors.len();
    if sn == 0 {
        return;
    }
    // counting sort survivors by candidate length → contiguous length runs
    b.count.clear();
    b.count.resize(max_len as usize + 1, 0);
    for k in 0..sn {
        b.count[c.lens[b.survivors[k] as usize] as usize] += 1;
    }
    let mut acc = 0u32;
    for x in b.count.iter_mut() {
        let t = *x;
        *x = acc;
        acc += t;
    }
    if b.surv_sorted.len() < sn {
        b.surv_sorted.resize(sn, 0);
    }
    for k in 0..sn {
        let si = b.survivors[k];
        let ln = c.lens[si as usize] as usize;
        let pos = b.count[ln] as usize;
        b.count[ln] += 1;
        b.surv_sorted[pos] = si;
    }
    // disjoint borrows: read surv_sorted, write hits + tile
    let Bufs {
        surv_sorted,
        hits,
        tile,
        ..
    } = &mut *b;
    let surv = &surv_sorted[..sn];
    let mut p = 0usize;
    while p < sn {
        let l = c.lens[surv[p] as usize] as usize;
        let mut q = p;
        while q < sn && c.lens[surv[q] as usize] as usize == l {
            q += 1;
        }
        score_run(&surv[p..q], l, &c.blob, &c.offs, lc_needle, use_simd, tile, hits);
        p = q;
    }
}

fn prepared_run(
    c: &Ctx,
    b: &mut Bufs,
    needle: &[u8],
    use_simd: bool,
    boosts: Option<&[i32]>,
    out_i: &mut [i32],
    out_s: &mut [i32],
) -> i32 {
    b.lc_needle.clear();
    b.lc_needle.extend(needle.iter().map(|&x| lower(x)));
    b.hits.clear();
    let nlen = b.lc_needle.len();
    let cap = out_i.len();
    if nlen == 0 {
        // empty query = source order (no scoring), so boosts do not reorder here — they refine
        // the SCORED ranking; an unqueried menu keeps its server/sort-text order.
        let k = cap.min(c.n);
        for i in 0..k {
            out_i[i] = i as i32;
            out_s[i] = 0;
        }
        return k as i32;
    }
    // needle signature = OR of its bytes' bits; a candidate can match only if its
    // signature is a SUPERSET (needle & cand == needle).
    let mut nsig = 0u32;
    for &x in &b.lc_needle {
        nsig |= sig_byte(x);
    }
    // stable raw view of lc_needle (not resized below) so we can also mutate b
    let lc_needle: &[u8] = unsafe { slice::from_raw_parts(b.lc_needle.as_ptr(), nlen) };

    scan_range(c, b, lc_needle, nsig, use_simd, 0, c.n);
    apply_boosts(&mut b.hits, boosts);
    radix_finalize(b, out_i, out_s)
}

/// PARALLEL variant of `prepared_run` (ABI 4): the per-candidate scoring is embarrassingly parallel, so split
/// `[0, n)` into contiguous per-thread blocks scanned CONCURRENTLY on a scoped thread pool, then merge each
/// block's hits IN THREAD ORDER and radix-finalize once. This is what lets a huge-pool match (single-threaded
/// ~250 ms at ~2M candidates) finish in tens of ms — fast enough to keep up with typing.
///
/// Byte-identical to the sequential `prepared_run`: thread `t` owns the index block `[t*chunk, (t+1)*chunk)`;
/// within a block `scan_range` already leaves each `(score, len)` tie in index-ascending order, and because
/// block `t`'s indices all precede block `t+1`'s, merging blocks in ascending `t` keeps the global order
/// index-ascending — so the stable radix breaks ties exactly as the single-thread scan. `c` is immutable
/// during a match, shared across the scoped threads by reference (no copy). Small pools / single core fall
/// back to the sequential path (thread setup + merge is not worth it there).
fn parallel_prepared_run(
    c: &Ctx,
    b: &mut Bufs,
    needle: &[u8],
    use_simd: bool,
    out_i: &mut [i32],
    out_s: &mut [i32],
) -> (i32, i32) {
    b.lc_needle.clear();
    b.lc_needle.extend(needle.iter().map(|&x| lower(x)));
    let nlen = b.lc_needle.len();
    let cap = out_i.len();
    if nlen == 0 {
        // empty query = every candidate "matches" → the TRUE total is the whole pool, of which the first
        // `cap` are returned in source order.
        let k = cap.min(c.n);
        for i in 0..k {
            out_i[i] = i as i32;
            out_s[i] = 0;
        }
        return (k as i32, c.n as i32);
    }
    let mut nsig = 0u32;
    for &x in &b.lc_needle {
        nsig |= sig_byte(x);
    }
    let n = c.n;
    // Parallelise only a large pool; cap the fan-out so merge/spawn overhead stays bounded on many-core boxes.
    const PAR_MIN: usize = 40_000;
    const MAX_THREADS: usize = 16;
    let hw = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1);
    let nthreads = hw.min(MAX_THREADS);
    if nthreads <= 1 || n < PAR_MIN {
        b.hits.clear();
        let lc_needle: &[u8] = unsafe { slice::from_raw_parts(b.lc_needle.as_ptr(), nlen) };
        scan_range(c, b, lc_needle, nsig, use_simd, 0, n);
        let total = b.hits.len() as i32; // ALL matches (before the cap radix_finalize applies)
        return (radix_finalize(b, out_i, out_s), total);
    }
    // Own the needle in a local (a handful of bytes) so nothing borrows `b` across the scope → `b` is free for
    // the merge + finalize.
    let lc_owned: Vec<u8> = b.lc_needle.clone();
    let chunk = n.div_ceil(nthreads);
    let mut merged: Vec<Hit> = Vec::new();
    std::thread::scope(|s| {
        let lc: &[u8] = &lc_owned;
        let mut handles = Vec::with_capacity(nthreads);
        for t in 0..nthreads {
            let start = (t * chunk).min(n);
            let end = ((t + 1) * chunk).min(n);
            if start >= end {
                break;
            }
            handles.push(s.spawn(move || {
                let mut lb = Bufs::new();
                scan_range(c, &mut lb, lc, nsig, use_simd, start, end);
                lb.hits
            }));
        }
        for h in handles {
            merged.extend(h.join().unwrap_or_default()); // thread order = index order → tie-break preserved
        }
    });
    b.hits = merged;
    let total = b.hits.len() as i32; // ALL matches (before the cap)
    (radix_finalize(b, out_i, out_s), total)
}

/// Match against the prepared context: needle-only, one coarse call per keystroke
/// (AVX2 inter-candidate SIMD where available). The O(1) signature prefilter skips
/// candidates lacking a needle byte with no scan.
///
/// # Safety
/// `needle` valid for `needle_len`; out arrays hold `max_out` elements. Must be
/// called after `lvimfuzzy_prepare` on the same thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lvimfuzzy_match_prepared(
    needle: *const u8,
    needle_len: i32,
    out_idx: *mut i32,
    out_score: *mut i32,
    max_out: i32,
) -> i32 {
    unsafe { match_prepared_impl(needle, needle_len, out_idx, out_score, max_out, true) }
}

/// Scalar-forced variant (SIMD disabled) — used by the correctness diff test /
/// bench parity column.
///
/// # Safety
/// Same contract as `lvimfuzzy_match_prepared`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lvimfuzzy_match_prepared_scalar(
    needle: *const u8,
    needle_len: i32,
    out_idx: *mut i32,
    out_score: *mut i32,
    max_out: i32,
) -> i32 {
    unsafe { match_prepared_impl(needle, needle_len, out_idx, out_score, max_out, false) }
}

/// PARALLEL prepared match (ABI 4): same contract as `lvimfuzzy_match_prepared`, but the scan is fanned out
/// across worker threads (AVX2 per thread). Fast enough at millions of candidates to run per keystroke without
/// blocking typing. Byte-identical results to the single-threaded entry.
///
/// # Safety
/// `needle` valid for `needle_len`; out arrays hold `max_out` elements. Called after `lvimfuzzy_prepare` on the
/// same thread (the scoped worker threads only READ the context, which is immutable during a match).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lvimfuzzy_match_prepared_par(
    needle: *const u8,
    needle_len: i32,
    out_idx: *mut i32,
    out_score: *mut i32,
    max_out: i32,
) -> i32 {
    if needle.is_null() || out_idx.is_null() || out_score.is_null() || max_out <= 0 {
        return 0;
    }
    unsafe {
        let needle = slice::from_raw_parts(needle, needle_len.max(0) as usize);
        let cap = max_out as usize;
        let out_i = slice::from_raw_parts_mut(out_idx, cap);
        let out_s = slice::from_raw_parts_mut(out_score, cap);
        CTX.with(|ccell| {
            let c = ccell.borrow();
            BUFS.with(|bcell| parallel_prepared_run(&c, &mut bcell.borrow_mut(), needle, true, out_i, out_s).0)
        })
    }
}

/// # Safety
/// See `lvimfuzzy_match_prepared`.
unsafe fn match_prepared_impl(
    needle: *const u8,
    needle_len: i32,
    out_idx: *mut i32,
    out_score: *mut i32,
    max_out: i32,
    use_simd: bool,
) -> i32 {
    if needle.is_null() || out_idx.is_null() || out_score.is_null() || max_out <= 0 {
        return 0;
    }
    unsafe {
        let needle = slice::from_raw_parts(needle, needle_len.max(0) as usize);
        let cap = max_out as usize;
        let out_i = slice::from_raw_parts_mut(out_idx, cap);
        let out_s = slice::from_raw_parts_mut(out_score, cap);
        CTX.with(|ccell| {
            let c = ccell.borrow();
            BUFS.with(|bcell| {
                prepared_run(&c, &mut bcell.borrow_mut(), needle, use_simd, None, out_i, out_s)
            })
        })
    }
}

/// BOOSTED prepared match (ABI 6): same single-threaded prepared scan as
/// `lvimfuzzy_match_prepared`, but a per-candidate `boosts` array (length = the prepared
/// candidate count) is added to each match score before ranking. Used by lvim-cmp to fold
/// exact-prefix / proximity / frecency signals into the fuzzy order. A null / empty boosts
/// pointer is identical to the plain entry point.
///
/// # Safety
/// `needle` valid for `needle_len`; `boosts` valid for `boosts_len` (or null); out arrays
/// hold `max_out` elements. Called after `lvimfuzzy_prepare` on the same thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lvimfuzzy_match_prepared_boost(
    needle: *const u8,
    needle_len: i32,
    boosts: *const i32,
    boosts_len: i32,
    out_idx: *mut i32,
    out_score: *mut i32,
    max_out: i32,
) -> i32 {
    if needle.is_null() || out_idx.is_null() || out_score.is_null() || max_out <= 0 {
        return 0;
    }
    unsafe {
        let needle = slice::from_raw_parts(needle, needle_len.max(0) as usize);
        let cap = max_out as usize;
        let out_i = slice::from_raw_parts_mut(out_idx, cap);
        let out_s = slice::from_raw_parts_mut(out_score, cap);
        let boosts = if boosts.is_null() || boosts_len <= 0 {
            None
        } else {
            Some(slice::from_raw_parts(boosts, boosts_len as usize))
        };
        CTX.with(|ccell| {
            let c = ccell.borrow();
            BUFS.with(|bcell| prepared_run(&c, &mut bcell.borrow_mut(), needle, true, boosts, out_i, out_s))
        })
    }
}

// ─── cooperative chunked match (ABI 3): drive one keystroke's match in slices ─
/// Begin a chunked match of `needle` over the prepared context: record the lowercased needle + signature,
/// reset the cursor, clear the hit accumulator. Returns the candidate count `n` (the caller steps `[0, n)`).
///
/// # Safety
/// `needle` valid for `needle_len`. Called after `lvimfuzzy_prepare` on the same thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lvimfuzzy_match_chunk_begin(needle: *const u8, needle_len: i32) -> i32 {
    if needle.is_null() {
        return 0;
    }
    let needle = unsafe { slice::from_raw_parts(needle, needle_len.max(0) as usize) };
    BUFS.with(|bc| bc.borrow_mut().hits.clear());
    let n = CTX.with(|cc| cc.borrow().n);
    CHUNK.with(|cc| {
        let mut ch = cc.borrow_mut();
        ch.lc_needle.clear();
        ch.lc_needle.extend(needle.iter().map(|&x| lower(x)));
        let mut nsig = 0u32;
        for &x in &ch.lc_needle {
            nsig |= sig_byte(x);
        }
        ch.nsig = nsig;
        ch.cursor = 0;
        ch.n = n;
    });
    n as i32
}

/// Scan the next up-to-`budget` candidates (appending their hits to the accumulator); returns the new cursor.
/// The match is complete once the returned cursor == the `n` from `begin`. An empty needle scans nothing
/// (finish returns the source-order head). Same thread as `begin`.
#[unsafe(no_mangle)]
pub extern "C" fn lvimfuzzy_match_chunk_step(budget: i32) -> i32 {
    let budget = budget.max(1) as usize;
    // The CURRENT context size — clamp against it too. It may differ from the `n` recorded at begin: it GROWS
    // under a stream-append (fine — we scan the begin-time head), or SHRINKS/empties if the context was freed
    // or replaced between slices. The Lua driver aborts a stale slice before calling us, but clamp defensively
    // so a race can never index the context out of bounds; if it shrank below `n`, report done so the caller
    // stops (its result is discarded upstream).
    let cur_n = CTX.with(|cc| cc.borrow().n);
    let (start, end, nsig, lc): (usize, usize, u32, Vec<u8>) = CHUNK.with(|cc| {
        let ch = cc.borrow();
        if ch.lc_needle.is_empty() {
            (ch.n, ch.n, 0, Vec::new())
        } else {
            let start = ch.cursor.min(cur_n);
            let end = (start + budget).min(ch.n).min(cur_n);
            (start, end, ch.nsig, ch.lc_needle.clone())
        }
    });
    if !lc.is_empty() && start < end {
        CTX.with(|ccell| {
            let c = ccell.borrow();
            BUFS.with(|bc| scan_range(&c, &mut bc.borrow_mut(), &lc, nsig, true, start, end));
        });
    }
    CHUNK.with(|cc| {
        let mut ch = cc.borrow_mut();
        // context shrank/replaced under us → nothing valid left; jump to `n` so the driver sees "done"
        ch.cursor = if cur_n < ch.n { ch.n } else { end };
        ch.cursor as i32
    })
}

/// Finalize the chunked match: radix-sort the accumulated hits into the caller's out arrays (capacity
/// `max_out`), return the count. An empty needle yields the first `max_out` in source order. Byte-identical
/// to `match_prepared` for the same needle + context.
///
/// # Safety
/// `out_idx`/`out_score` hold `max_out` elements. Same thread as `begin`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lvimfuzzy_match_chunk_finish(out_idx: *mut i32, out_score: *mut i32, max_out: i32) -> i32 {
    if out_idx.is_null() || out_score.is_null() || max_out <= 0 {
        return 0;
    }
    let cap = max_out as usize;
    let out_i = unsafe { slice::from_raw_parts_mut(out_idx, cap) };
    let out_s = unsafe { slice::from_raw_parts_mut(out_score, cap) };
    let empty = CHUNK.with(|cc| cc.borrow().lc_needle.is_empty());
    if empty {
        let n = CTX.with(|cc| cc.borrow().n);
        let k = cap.min(n);
        for i in 0..k {
            out_i[i] = i as i32;
            out_s[i] = 0;
        }
        return k as i32;
    }
    BUFS.with(|bc| radix_finalize(&mut bc.borrow_mut(), out_i, out_s))
}

// ─── C ABI (loaded from Lua via LuaJIT FFI) ──────────────────────────────────
/// ABI version. The base contract (prepare / match_prepared / match / avx2) is
/// ABI ≥ 1; incremental `append` + `ctx_free` were added in ABI 2; the cooperative
/// chunked match (`match_chunk_begin`/`step`/`finish`) in ABI 3; the PARALLEL match
/// (`match_prepared_par`) in ABI 4; the handle-addressed BLOB-ingest contexts
/// (`ctx_new`/`ctx_append_blob`/`ctx_flush`/`ctx_count`/`ctx_text`/`ctx_match_par`/`ctx_drop`)
/// in ABI 5. The loader accepts any ABI ≥ 1 (existing consumers keep working on this newer
/// .so) and exposes each tier's features only when this reports at or above it.
#[unsafe(no_mangle)]
pub extern "C" fn lvimfuzzy_abi() -> i32 {
    6
}

/// 1 if the running CPU has AVX2 (the inter-candidate SIMD tile path is active),
/// else 0 (the scalar path runs). For :checkhealth reporting.
#[unsafe(no_mangle)]
pub extern "C" fn lvimfuzzy_avx2() -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            1
        } else {
            0
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        0
    }
}

/// Match `n` candidates (blob + offsets) against `needle`; fill caller-owned
/// out_idx/out_score (capacity max_out) in ranked order, return the count.
///
/// # Safety
/// Pointers must be valid for the stated lengths (offsets has n+1 elements,
/// blob length == offsets[n], out arrays hold max_out elements).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lvimfuzzy_match(
    hay: *const u8,
    offs: *const i32,
    n: i32,
    needle: *const u8,
    needle_len: i32,
    out_idx: *mut i32,
    out_score: *mut i32,
    max_out: i32,
) -> i32 {
    if hay.is_null()
        || offs.is_null()
        || out_idx.is_null()
        || out_score.is_null()
        || n <= 0
        || max_out <= 0
    {
        return 0;
    }
    unsafe {
        let n = n as usize;
        let offsets = slice::from_raw_parts(offs, n + 1);
        let blob = slice::from_raw_parts(hay, offsets[n] as usize);
        let needle = slice::from_raw_parts(needle, needle_len.max(0) as usize);
        let cap = max_out as usize;
        let out_i = slice::from_raw_parts_mut(out_idx, cap);
        let out_s = slice::from_raw_parts_mut(out_score, cap);
        run_match(blob, offsets, n, needle, out_i, out_s)
    }
}

// ─── correctness: SIMD path must be byte-identical to the scalar path ─────────
#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic identifier-like candidate generator (mirrors the bench set:
    /// camelCase / snake_case / PascalCase / dotted, 1–4 stems, numeric suffix).
    fn gen_cands(n: usize) -> Vec<String> {
        const STEMS: [&str; 24] = [
            "get", "set", "update", "render", "buffer", "window", "client", "config", "handler",
            "match", "source", "complete", "fuzzy", "score", "index", "load", "parse", "token",
            "insert", "cursor", "menu", "snippet", "path", "async",
        ];
        let mut seed: u64 = 42;
        let mut rnd = |m: usize| {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345) % 2147483648;
            (seed as usize % m) + 1
        };
        let cap = |s: &str| {
            let mut c = s.chars();
            c.next()
                .map(|f| f.to_ascii_uppercase().to_string() + c.as_str())
                .unwrap_or_default()
        };
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let parts: Vec<&str> = (0..1 + rnd(3))
                .map(|_| STEMS[rnd(STEMS.len()) - 1])
                .collect();
            let mut s = match rnd(4) {
                1 => parts[0].to_string() + &parts[1..].iter().map(|p| cap(p)).collect::<String>(),
                2 => parts.join("_"),
                3 => parts.iter().map(|p| cap(p)).collect::<String>(),
                _ => parts.join("."),
            };
            if rnd(6) == 1 {
                s.push_str(&rnd(99).to_string());
            }
            out.push(s);
        }
        out
    }

    fn build_ctx(cands: &[String]) -> Ctx {
        let mut blob = Vec::new();
        let mut offs = vec![0i32];
        let mut sigs = Vec::new();
        let mut lens = Vec::new();
        for c in cands {
            let bytes = c.as_bytes();
            blob.extend_from_slice(bytes);
            offs.push(blob.len() as i32);
            let mut m = 0u32;
            for &bb in bytes {
                m |= sig_byte(bb);
            }
            sigs.push(m);
            lens.push(bytes.len() as u32);
        }
        Ctx {
            blob,
            offs,
            sigs,
            lens,
            n: cands.len(),
            pending: Vec::new(),
        }
    }

    /// Runs the SIMD and the scalar path over the WHOLE query set on both the 5k
    /// and 20k sets and asserts byte-identical (count + indices + scores) output.
    #[test]
    fn simd_matches_scalar() {
        let queries = [
            "f",
            "co",
            "buf",
            "conf",
            "gtcl",
            "handler",
            "updbuf",
            "asyncqueue",
            "get",
            "set",
            "client",
            "Client",
            "sc",
            "idx",
            "xyz",
            "zzzz",
            "cccc",
            "s",
            "parse",
            "cfg",
        ];
        for &size in &[5000usize, 20000] {
            let cands = gen_cands(size);
            let ctx = build_ctx(&cands);
            let mut b1 = Bufs::new();
            let mut b2 = Bufs::new();
            for q in queries {
                let cap = ctx.n;
                let (mut i1, mut s1) = (vec![0i32; cap], vec![0i32; cap]);
                let (mut i2, mut s2) = (vec![0i32; cap], vec![0i32; cap]);
                let n1 = prepared_run(&ctx, &mut b1, q.as_bytes(), true, None, &mut i1, &mut s1) as usize;
                let n2 =
                    prepared_run(&ctx, &mut b2, q.as_bytes(), false, None, &mut i2, &mut s2) as usize;
                assert_eq!(n1, n2, "match COUNT differs (size {size}, query {q:?})");
                assert_eq!(
                    &i1[..n1],
                    &i2[..n2],
                    "INDEX order differs (size {size}, query {q:?})"
                );
                assert_eq!(
                    &s1[..n1],
                    &s2[..n2],
                    "SCORES differ (size {size}, query {q:?})"
                );
            }
        }
    }

    /// Concatenated bytes + (n+1) offsets for a candidate slice (the shape the
    /// FFI prepare/append receive).
    fn blob_offs(cands: &[String]) -> (Vec<u8>, Vec<i32>) {
        let mut blob = Vec::new();
        let mut offs = vec![0i32];
        for c in cands {
            blob.extend_from_slice(c.as_bytes());
            offs.push(blob.len() as i32);
        }
        (blob, offs)
    }

    /// prepare(A) then append(B) must yield a context whose SoA arrays AND whose
    /// match() output are byte-identical to prepare(A ++ B), for every split and
    /// every query.
    #[test]
    fn append_equals_prepare_whole() {
        let queries = [
            "f", "co", "buf", "conf", "gtcl", "handler", "get", "s", "cs", "cfg", "zzzz", "client",
        ];
        let all = gen_cands(6000);
        for split in [0usize, 1, 17, 256, 3000, 5999, 6000] {
            // incremental: empty ctx, append A, then append B
            let mut inc = Ctx::new();
            let (ab, ao) = blob_offs(&all[..split]);
            if split > 0 {
                ctx_append(&mut inc, &ab, &ao, split);
            }
            let (bb, bo) = blob_offs(&all[split..]);
            if split < all.len() {
                ctx_append(&mut inc, &bb, &bo, all.len() - split);
            }
            // whole
            let whole = build_ctx(&all);
            // SoA identity
            assert_eq!(inc.n, whole.n, "n (split {split})");
            assert_eq!(inc.blob, whole.blob, "blob (split {split})");
            assert_eq!(inc.offs, whole.offs, "offs (split {split})");
            assert_eq!(inc.sigs, whole.sigs, "sigs (split {split})");
            assert_eq!(inc.lens, whole.lens, "lens (split {split})");
            // match-output identity for every query
            let mut b1 = Bufs::new();
            let mut b2 = Bufs::new();
            for q in queries {
                let cap = whole.n;
                let (mut i1, mut s1) = (vec![0i32; cap], vec![0i32; cap]);
                let (mut i2, mut s2) = (vec![0i32; cap], vec![0i32; cap]);
                let n1 = prepared_run(&inc, &mut b1, q.as_bytes(), true, None, &mut i1, &mut s1) as usize;
                let n2 =
                    prepared_run(&whole, &mut b2, q.as_bytes(), true, None, &mut i2, &mut s2) as usize;
                assert_eq!(n1, n2, "count q={q:?} split={split}");
                assert_eq!(&i1[..n1], &i2[..n2], "idx q={q:?} split={split}");
                assert_eq!(&s1[..n1], &s2[..n2], "score q={q:?} split={split}");
            }
        }
    }

    /// The chunked path (`scan_range` over `[0,n)` in `budget`-sized slices, then `radix_finalize`) — what the
    /// ABI-3 begin/step/finish FFI drives across event-loop ticks — mimicked here on an explicit ctx/bufs.
    fn chunked_run(
        c: &Ctx,
        b: &mut Bufs,
        needle: &[u8],
        use_simd: bool,
        budget: usize,
        out_i: &mut [i32],
        out_s: &mut [i32],
    ) -> i32 {
        b.lc_needle.clear();
        b.lc_needle.extend(needle.iter().map(|&x| lower(x)));
        b.hits.clear();
        let nlen = b.lc_needle.len();
        if nlen == 0 {
            let k = out_i.len().min(c.n);
            for i in 0..k {
                out_i[i] = i as i32;
                out_s[i] = 0;
            }
            return k as i32;
        }
        let mut nsig = 0u32;
        for &x in &b.lc_needle {
            nsig |= sig_byte(x);
        }
        let lc: &[u8] = unsafe { slice::from_raw_parts(b.lc_needle.as_ptr(), nlen) };
        let mut cursor = 0usize;
        while cursor < c.n {
            let end = (cursor + budget.max(1)).min(c.n);
            scan_range(c, b, lc, nsig, use_simd, cursor, end);
            cursor = end;
        }
        radix_finalize(b, out_i, out_s)
    }

    /// The cooperative CHUNKED match, at every budget from a single candidate to the whole set, is
    /// BYTE-IDENTICAL (count + indices + scores) to the single-shot `prepared_run` — for every query,
    /// incl. empty / broad / selective / no-match. Proves the async path can never reorder results.
    #[test]
    fn chunked_equals_whole() {
        let queries = [
            "", "f", "co", "buf", "conf", "gtcl", "handler", "updbuf", "asyncqueue", "e", "get", "z",
            "zzz",
        ];
        for &n in &[5000usize, 20000] {
            let cands = gen_cands(n);
            let c = build_ctx(&cands);
            let mut b1 = Bufs::new();
            let mut b2 = Bufs::new();
            for q in queries {
                let cap = 1000usize;
                let (mut wi, mut ws) = (vec![0i32; cap], vec![0i32; cap]);
                let wn = prepared_run(&c, &mut b1, q.as_bytes(), true, None, &mut wi, &mut ws) as usize;
                for &budget in &[1usize, 7, 128, 1000, 99_999, n] {
                    let (mut ci, mut cs) = (vec![0i32; cap], vec![0i32; cap]);
                    let cn = chunked_run(&c, &mut b2, q.as_bytes(), true, budget, &mut ci, &mut cs) as usize;
                    assert_eq!(wn, cn, "count q={q:?} n={n} budget={budget}");
                    assert_eq!(&wi[..wn], &ci[..cn], "idx q={q:?} n={n} budget={budget}");
                    assert_eq!(&ws[..wn], &cs[..cn], "score q={q:?} n={n} budget={budget}");
                }
            }
        }
    }

    /// The PARALLEL prepared match is BYTE-IDENTICAL (count + indices + scores) to the single-threaded scan for
    /// every query — tested both BELOW the parallelisation threshold (sequential fallback) and well ABOVE it
    /// (the real multi-thread fan-out + in-thread-order merge), so a mis-ordered merge would break ties and be
    /// caught here.
    #[test]
    fn parallel_equals_whole() {
        let queries = [
            "", "f", "co", "buf", "conf", "gtcl", "handler", "updbuf", "asyncqueue", "e", "get", "z", "zzz", "s",
            "cccc", "parse",
        ];
        for &n in &[5000usize, 50_000, 120_000] {
            let cands = gen_cands(n);
            let c = build_ctx(&cands);
            let mut b1 = Bufs::new();
            let mut b2 = Bufs::new();
            for q in queries {
                let cap = 1000usize;
                let (mut wi, mut ws) = (vec![0i32; cap], vec![0i32; cap]);
                let wn = prepared_run(&c, &mut b1, q.as_bytes(), true, None, &mut wi, &mut ws) as usize;
                let (mut pi, mut ps) = (vec![0i32; cap], vec![0i32; cap]);
                let (pw, ptot) = parallel_prepared_run(&c, &mut b2, q.as_bytes(), true, &mut pi, &mut ps);
                let pn = pw as usize;
                assert_eq!(wn, pn, "count q={q:?} n={n}");
                assert_eq!(&wi[..wn], &pi[..pn], "idx q={q:?} n={n}");
                assert_eq!(&ws[..wn], &ps[..pn], "score q={q:?} n={n}");
                // the TRUE total (before the 1000 cap) is ≥ the written count, and equals it when uncapped
                assert!(ptot as usize >= pn, "total < written q={q:?} n={n}");
                if pn < cap {
                    assert_eq!(ptot as usize, pn, "total ≠ matches when uncapped q={q:?} n={n}");
                }
            }
        }
    }

    /// Ingest `lines` into a fresh Ctx as a raw '\n'-joined BLOB, fed in `chunk`-sized byte slices (so a chunk
    /// boundary can fall MID-LINE — the pending-line carry is exercised), optionally without a trailing '\n'
    /// (the flush path), mirroring `lvimfuzzy_ctx_append_blob` + `lvimfuzzy_ctx_flush`.
    fn feed_blob(lines: &[String], chunk: usize, trailing_nl: bool) -> Ctx {
        let mut buf = Vec::new();
        for (k, l) in lines.iter().enumerate() {
            buf.extend_from_slice(l.as_bytes());
            if trailing_nl || k + 1 < lines.len() {
                buf.push(b'\n');
            }
        }
        let mut c = Ctx::new();
        let mut i = 0;
        while i < buf.len() {
            let e = (i + chunk.max(1)).min(buf.len());
            ctx_append_blob(&mut c, &buf[i..e]);
            i = e;
        }
        ctx_flush_blob(&mut c);
        c
    }

    /// A blob-ingested context — fed the candidate lines as raw bytes in chunks of every size (incl. 1 byte,
    /// so every boundary splits a line) with and without a trailing newline — is byte-identical (SoA arrays
    /// AND match output for every query) to preparing the same lines the offsets-based way. This is what lets
    /// the picker hand fd's stdout straight to the native context with ZERO Lua per-row work.
    #[test]
    fn blob_equals_prepare_whole() {
        let queries = [
            "", "f", "co", "buf", "conf", "gtcl", "handler", "get", "s", "cs", "cfg", "zzzz", "client", "parse",
        ];
        let all = gen_cands(4000);
        let whole = build_ctx(&all);
        for &chunk in &[1usize, 2, 7, 64, 500, 4096, usize::MAX] {
            for &trailing in &[true, false] {
                let inc = feed_blob(&all, chunk, trailing);
                assert_eq!(inc.n, whole.n, "n (chunk {chunk}, trailing {trailing})");
                assert_eq!(inc.blob, whole.blob, "blob (chunk {chunk}, trailing {trailing})");
                assert_eq!(inc.offs, whole.offs, "offs (chunk {chunk}, trailing {trailing})");
                assert_eq!(inc.sigs, whole.sigs, "sigs (chunk {chunk}, trailing {trailing})");
                assert_eq!(inc.lens, whole.lens, "lens (chunk {chunk}, trailing {trailing})");
                let mut b1 = Bufs::new();
                let mut b2 = Bufs::new();
                for q in queries {
                    let cap = 1000usize;
                    let (mut i1, mut s1) = (vec![0i32; cap], vec![0i32; cap]);
                    let (mut i2, mut s2) = (vec![0i32; cap], vec![0i32; cap]);
                    let n1 = prepared_run(&inc, &mut b1, q.as_bytes(), true, None, &mut i1, &mut s1) as usize;
                    let n2 = prepared_run(&whole, &mut b2, q.as_bytes(), true, None, &mut i2, &mut s2) as usize;
                    assert_eq!(n1, n2, "count q={q:?} chunk={chunk}");
                    assert_eq!(&i1[..n1], &i2[..n2], "idx q={q:?} chunk={chunk}");
                    assert_eq!(&s1[..n1], &s2[..n2], "score q={q:?} chunk={chunk}");
                }
            }
        }
    }

    /// `ctx_text`-shape read-back: every candidate's stored bytes equal the ingested line, at its 1-based
    /// index (start = offs[idx-1], end = offs[idx]) — the accessor the picker renders visible rows through.
    #[test]
    fn blob_text_roundtrip() {
        let all = gen_cands(1500);
        let c = feed_blob(&all, 5, true);
        assert_eq!(c.n, all.len());
        for (k, line) in all.iter().enumerate() {
            let s = c.offs[k] as usize;
            let e = c.offs[k + 1] as usize;
            assert_eq!(&c.blob[s..e], line.as_bytes(), "text roundtrip idx {k}");
        }
    }

    /// Empty lines in the byte stream are skipped (a fuzzy candidate is never the empty string) — a blob with
    /// blank lines ingests only the non-empty ones, matching the finders' `p ~= ""` guard.
    #[test]
    fn blob_skips_empty_lines() {
        let mut c = Ctx::new();
        ctx_append_blob(&mut c, b"alpha\n\nbeta\n\n\ngamma\n");
        ctx_flush_blob(&mut c);
        assert_eq!(c.n, 3);
        let read = |i: usize| {
            let s = c.offs[i] as usize;
            let e = c.offs[i + 1] as usize;
            std::str::from_utf8(&c.blob[s..e]).unwrap().to_string()
        };
        assert_eq!(read(0), "alpha");
        assert_eq!(read(1), "beta");
        assert_eq!(read(2), "gamma");
    }

    /// Boosts fold into the score before ranking: a null / all-zero boost is byte-identical
    /// to the plain match, and a large positive boost promotes its candidate to the top
    /// without changing the match SET.
    #[test]
    fn boost_reorders_and_zero_is_plain() {
        let cands: Vec<String> =
            vec!["getBuffer".into(), "bunchOfStuff".into(), "abufoo".into(), "cfg".into()];
        let ctx = build_ctx(&cands);
        let q = b"buf";
        let cap = ctx.n;

        // plain (None) == all-zeros boost, in both index order and scores
        let mut bp = Bufs::new();
        let (mut ip, mut sp) = (vec![0i32; cap], vec![0i32; cap]);
        let np = prepared_run(&ctx, &mut bp, q, true, None, &mut ip, &mut sp) as usize;

        let zeros = vec![0i32; ctx.n];
        let mut bz = Bufs::new();
        let (mut iz, mut sz) = (vec![0i32; cap], vec![0i32; cap]);
        let nz = prepared_run(&ctx, &mut bz, q, true, Some(&zeros), &mut iz, &mut sz) as usize;
        assert_eq!(&ip[..np], &iz[..nz], "zero boosts must match the plain order");
        assert_eq!(&sp[..np], &sz[..nz], "zero boosts must match the plain scores");

        // hugely boost the LAST-ranked match → it must rank first, set unchanged
        assert!(np >= 2, "need at least two matches for the reorder check");
        let underdog = ip[np - 1];
        let mut boosts = vec![0i32; ctx.n];
        boosts[underdog as usize] = 10_000;
        let mut bb = Bufs::new();
        let (mut ib, mut sb) = (vec![0i32; cap], vec![0i32; cap]);
        let nb = prepared_run(&ctx, &mut bb, q, true, Some(&boosts), &mut ib, &mut sb) as usize;
        assert_eq!(nb, np, "a boost must not change the match SET (same count)");
        assert_eq!(ib[0], underdog, "a huge boost must rank the boosted candidate first");
    }
}
