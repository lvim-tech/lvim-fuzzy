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
}

impl Ctx {
    const fn new() -> Self {
        Ctx {
            blob: Vec::new(),
            offs: Vec::new(),
            sigs: Vec::new(),
            lens: Vec::new(),
            n: 0,
        }
    }
}

thread_local! {
    static CTX: RefCell<Ctx> = const { RefCell::new(Ctx::new()) };
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
fn prepared_run(
    c: &Ctx,
    b: &mut Bufs,
    needle: &[u8],
    use_simd: bool,
    out_i: &mut [i32],
    out_s: &mut [i32],
) -> i32 {
    b.lc_needle.clear();
    b.lc_needle.extend(needle.iter().map(|&x| lower(x)));
    b.hits.clear();
    let nlen = b.lc_needle.len();
    let cap = out_i.len();
    if nlen == 0 {
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

    if nlen == 1 {
        let want = lc_needle[0];
        for i in 0..c.n {
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
    } else {
        // Collect survivors: O(1) signature reject, then a cheap subsequence
        // confirm (rejects wrong-order candidates the DP would only reject after a
        // full score — pruning them shrinks the SIMD tile count for broad queries;
        // for selective queries the survivor set is tiny so this is nearly free).
        b.survivors.clear();
        let mut max_len = 0u32;
        for i in 0..c.n {
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
        if sn > 0 {
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
                score_run(
                    &surv[p..q],
                    l,
                    &c.blob,
                    &c.offs,
                    lc_needle,
                    use_simd,
                    tile,
                    hits,
                );
                p = q;
            }
        }
    }
    radix_finalize(b, out_i, out_s)
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
                prepared_run(&c, &mut bcell.borrow_mut(), needle, use_simd, out_i, out_s)
            })
        })
    }
}

// ─── C ABI (loaded from Lua via LuaJIT FFI) ──────────────────────────────────
/// ABI version. The base contract (prepare / match_prepared / match / avx2) is
/// ABI ≥ 1; incremental `append` + `ctx_free` were added in ABI 2. The loader
/// accepts any ABI ≥ 1 (existing consumers keep working on this newer .so) and
/// exposes the ABI-2 features only when this reports ≥ 2.
#[unsafe(no_mangle)]
pub extern "C" fn lvimfuzzy_abi() -> i32 {
    2
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
                let n1 = prepared_run(&ctx, &mut b1, q.as_bytes(), true, &mut i1, &mut s1) as usize;
                let n2 =
                    prepared_run(&ctx, &mut b2, q.as_bytes(), false, &mut i2, &mut s2) as usize;
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
                let n1 = prepared_run(&inc, &mut b1, q.as_bytes(), true, &mut i1, &mut s1) as usize;
                let n2 =
                    prepared_run(&whole, &mut b2, q.as_bytes(), true, &mut i2, &mut s2) as usize;
                assert_eq!(n1, n2, "count q={q:?} split={split}");
                assert_eq!(&i1[..n1], &i2[..n2], "idx q={q:?} split={split}");
                assert_eq!(&s1[..n1], &s2[..n2], "score q={q:?} split={split}");
            }
        }
    }
}
