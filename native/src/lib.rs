//! lvim-fuzzy native library: the C ABI that Neovim (LuaJIT FFI) uses to reach
//! lvim-fuzzy-core, the search engine shared with lvim-fuzzy-cli. Nothing of the
//! matching lives here: the scoring model, the query syntax, the search and the
//! grep scopes are core's, so a query ranks the same in lvim-picker, lvim-cmp and
//! the terminal finder.
//!
//! What this layer adds:
//! - contexts addressed by a handle (a candidate set that lives here, never as Lua
//!   strings), filled from a Lua list (`ctx_push`) or from a process's raw output
//!   (`ctx_append_blob`);
//! - narrowing: while the query only grows, the next search reads only the
//!   previous matches (and whatever was added since);
//! - boosts: per-candidate score offsets folded in before the ranking (lvim-cmp);
//! - positions for highlighting, as byte columns or as character indices.
//!
//! Every entry runs on Neovim's main thread; the search itself fans out over
//! every core inside core.

use std::cell::RefCell;
use std::collections::HashMap;
use std::os::raw::c_char;
use std::slice;

use lvim_fuzzy_core::{
    CaseMode, EXACT_WINDOW, Hit, Hits, MatchConfig, Options, Pattern, Scheme, Scope, Store,
    TYPOS_AUTO, Tiebreak, narrows, search,
};

/// The ABI this library implements. 7 = the first one on lvim-fuzzy-core; the
/// Lua side refuses anything else.
pub const ABI: i32 = 7;

/// Search options as Lua passes them (`lvimfuzzy_opts`). A null pointer means
/// the defaults (see [`Opts::default`]).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Opts {
    /// 0 whole candidate, 1 a grep result's line, 2 its path, 3 its file's content
    pub scope: i32,
    /// 0 smart, 1 ignore, 2 respect
    pub case_mode: i32,
    /// 1 = fzf's extended syntax (`'x ^x x$ !x a | b`, spaces = AND)
    pub extended: i32,
    /// 1 = every term is an exact substring
    pub exact: i32,
    /// 1 = Latin letters match their accented forms (`cafe` finds `café`)
    pub normalize: i32,
    /// 0 default, 1 path, 2 history
    pub scheme: i32,
    /// characters a fuzzy term may miss; -1 = by the term's length
    pub typos: i32,
    /// the widest window scored exactly; -1 = core's default, 0 = always exactly
    pub exact_window: i32,
}

impl Default for Opts {
    fn default() -> Opts {
        Opts {
            scope: 0,
            case_mode: 0,
            extended: 1,
            exact: 0,
            normalize: 1,
            scheme: 0,
            typos: 0,
            exact_window: -1,
        }
    }
}

impl Opts {
    /// # Safety
    /// `p` is null or points to a valid `Opts`.
    unsafe fn read(p: *const Opts) -> Opts {
        if p.is_null() {
            Opts::default()
        } else {
            unsafe { *p }
        }
    }

    fn config(&self) -> MatchConfig {
        MatchConfig {
            case: match self.case_mode {
                1 => CaseMode::Ignore,
                2 => CaseMode::Respect,
                _ => CaseMode::Smart,
            },
            extended: self.extended != 0,
            exact: self.exact != 0,
            normalize: self.normalize != 0,
            scheme: match self.scheme {
                1 => Scheme::Path,
                2 => Scheme::History,
                _ => Scheme::Default,
            },
            typos: if self.typos < 0 {
                TYPOS_AUTO
            } else {
                self.typos.min(TYPOS_AUTO as i32 - 1) as u8
            },
            exact_window: if self.exact_window < 0 {
                EXACT_WINDOW
            } else {
                self.exact_window as usize
            },
        }
    }

    fn scope(&self) -> Scope {
        match self.scope {
            1 => Scope::Line,
            2 => Scope::Path,
            3 => Scope::File,
            _ => Scope::All,
        }
    }

    fn options(&self) -> Options {
        Options {
            cfg: self.config(),
            sort: true,
            tac: false,
            tiebreak: vec![Tiebreak::Length],
            scope: self.scope(),
        }
    }
}

/// The previous search of a context, for narrowing the next one.
struct Last {
    query: String,
    opts: Opts,
    hits: Hits,
    /// the candidate count it searched; what was added since is searched too
    n: usize,
}

struct Ctx {
    store: Store,
    last: Option<Last>,
}

struct Registry {
    map: HashMap<i32, Ctx>,
    next: i32,
}

thread_local! {
    static REGISTRY: RefCell<Registry> = RefCell::new(Registry { map: HashMap::new(), next: 1 });
}

fn with_ctx<R>(id: i32, f: impl FnOnce(&mut Ctx) -> R) -> Option<R> {
    REGISTRY.with(|r| r.borrow_mut().map.get_mut(&id).map(f))
}

/// # Safety
/// `p` is null (then `len` is ignored) or points to `len` readable bytes.
unsafe fn bytes<'a>(p: *const c_char, len: i32) -> &'a [u8] {
    if p.is_null() || len <= 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(p as *const u8, len as usize) }
    }
}

fn text_of(b: &[u8]) -> std::borrow::Cow<'_, str> {
    String::from_utf8_lossy(b)
}

// ─── the ranked result ──────────────────────────────────────────────────────

/// The ranked hits after the boosts: `boosts[idx]` is added to candidate idx's
/// score and the list is re-sorted by score, stably, so equal scores keep core's
/// order (shorter first, then earlier). None = core's order as it is.
fn boosted(hits: &Hits, boosts: Option<&[i32]>) -> Option<Vec<Hit>> {
    let (Hits::List(v), Some(bo)) = (hits, boosts) else {
        return None;
    };
    let mut w: Vec<Hit> = v.as_ref().clone();
    for h in &mut w {
        if let Some(b) = bo.get(h.idx as usize) {
            h.score = h.score.saturating_add(*b);
        }
    }
    w.sort_by_key(|h| std::cmp::Reverse(h.score));
    Some(w)
}

/// Copy the top `out_idx.len()` hits out (0-based indices); the count written.
fn write_out(
    hits: &Hits,
    boosts: Option<&[i32]>,
    out_idx: &mut [i32],
    out_score: &mut [i32],
) -> usize {
    let cap = out_idx.len().min(out_score.len());
    if let Some(w) = boosted(hits, boosts) {
        let n = w.len().min(cap);
        for (k, h) in w.iter().take(n).enumerate() {
            out_idx[k] = h.idx as i32;
            out_score[k] = h.score;
        }
        return n;
    }
    match hits {
        Hits::List(v) => {
            let n = v.len().min(cap);
            for (k, h) in v.iter().take(n).enumerate() {
                out_idx[k] = h.idx as i32;
                out_score[k] = h.score;
            }
            n
        }
        Hits::All { .. } => {
            let n = hits.len().min(cap);
            for k in 0..n {
                out_idx[k] = hits.get(k).unwrap_or(0) as i32;
                out_score[k] = 0;
            }
            n
        }
    }
}

/// Search `c` for `query`, narrowing from the previous search when the query
/// only grew and the options are the same.
fn ctx_search(c: &mut Ctx, query: &str, opts: Opts) -> Hits {
    let o = opts.options();
    let n = c.store.len();
    let subset = c.last.as_ref().and_then(|l| {
        (l.opts == opts && l.n <= n && narrows(&l.query, query))
            .then(|| l.hits.indices().map(|ix| (ix, l.n)))
            .flatten()
    });
    let hits = search(
        &c.store,
        query,
        &o,
        None,
        subset.as_ref().map(|(ix, from)| (&ix[..], *from)),
    )
    .unwrap_or(Hits::All {
        count: 0,
        tac: false,
    });
    c.last = Some(Last {
        query: query.to_string(),
        opts,
        hits: hits.clone(),
        n,
    });
    hits
}

// ─── C ABI ──────────────────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub extern "C" fn lvimfuzzy_abi() -> i32 {
    ABI
}

/// 1 when the running CPU has AVX2 (core's SIMD paths are in use), else 0.
#[unsafe(no_mangle)]
pub extern "C" fn lvimfuzzy_avx2() -> i32 {
    lvim_fuzzy_core::simd::available() as i32
}

/// A new empty context; its handle (> 0). `skip_empty` = 1 drops empty
/// candidates (a stream of paths); 0 keeps them, so indices stay parallel to a
/// Lua list.
#[unsafe(no_mangle)]
pub extern "C" fn lvimfuzzy_ctx_new(skip_empty: i32) -> i32 {
    REGISTRY.with(|r| {
        let mut r = r.borrow_mut();
        let id = r.next;
        r.next = r.next.checked_add(1).unwrap_or(1);
        let mut store = Store::new();
        store.skip_empty = skip_empty != 0;
        r.map.insert(id, Ctx { store, last: None });
        id
    })
}

/// Add `n` candidates: `data` holds them back to back, `offs` their n + 1
/// offsets. Returns the new count (-1 for an unknown handle).
///
/// # Safety
/// `data` holds `offs[n]` bytes; `offs` holds `n + 1` ascending offsets.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lvimfuzzy_ctx_push(
    id: i32,
    data: *const c_char,
    offs: *const i32,
    n: i32,
) -> i32 {
    let n = n.max(0) as usize;
    let offs: &[i32] = if offs.is_null() {
        &[]
    } else {
        unsafe { slice::from_raw_parts(offs, n + 1) }
    };
    let total = offs.last().copied().unwrap_or(0);
    let data = unsafe { bytes(data, total) };
    with_ctx(id, |c| {
        let parts: Vec<&[u8]> = (0..n)
            .map(|i| &data[offs[i] as usize..offs[i + 1] as usize])
            .collect();
        let texts: Option<Vec<&str>> = parts.iter().map(|b| std::str::from_utf8(b).ok()).collect();
        match texts {
            Some(t) if !c.store.skip_empty => c.store.extend_par(&t),
            _ => parts.iter().for_each(|b| c.store.push(b)),
        }
        c.store.len() as i32
    })
    .unwrap_or(-1)
}

/// Add a chunk of newline-separated output (any chunking: a line cut at the end
/// waits for the next chunk or `ctx_flush`). Returns the new count.
///
/// # Safety
/// `data` holds `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lvimfuzzy_ctx_append_blob(id: i32, data: *const c_char, len: i32) -> i32 {
    let data = unsafe { bytes(data, len) };
    with_ctx(id, |c| {
        c.store.append(data, b'\n');
        c.store.len() as i32
    })
    .unwrap_or(-1)
}

/// The end of the stream: a last line without a newline. Returns the count.
#[unsafe(no_mangle)]
pub extern "C" fn lvimfuzzy_ctx_flush(id: i32) -> i32 {
    with_ctx(id, |c| {
        c.store.flush(b'\n');
        c.store.len() as i32
    })
    .unwrap_or(-1)
}

#[unsafe(no_mangle)]
pub extern "C" fn lvimfuzzy_ctx_count(id: i32) -> i32 {
    with_ctx(id, |c| c.store.len() as i32).unwrap_or(0)
}

/// The text of candidate `idx` (1-based), valid until the context changes (the
/// caller copies it at once). Null and `*out_len = 0` when out of range.
///
/// # Safety
/// `out_len` points to a writable int.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lvimfuzzy_ctx_text(id: i32, idx: i32, out_len: *mut i32) -> *const c_char {
    let r = with_ctx(id, |c| {
        let i = idx as i64 - 1;
        (i >= 0 && (i as usize) < c.store.len()).then(|| {
            let t = c.store.text(i as usize);
            (t.as_ptr() as *const c_char, t.len() as i32)
        })
    })
    .flatten();
    let (p, len) = r.unwrap_or((std::ptr::null(), 0));
    if !out_len.is_null() {
        unsafe { *out_len = len };
    }
    p
}

/// Rank context `id` for `query`: the top `cap` (0-based indices, scores) in
/// `out_idx` / `out_score`, the count written returned, every match counted in
/// `*out_total` (the whole set for an empty query). `boosts` (may be null)
/// holds `nboosts` offsets, one per candidate.
///
/// # Safety
/// Every pointer is valid for the length given with it; `opts` may be null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lvimfuzzy_ctx_match(
    id: i32,
    query: *const c_char,
    query_len: i32,
    opts: *const Opts,
    boosts: *const i32,
    nboosts: i32,
    out_idx: *mut i32,
    out_score: *mut i32,
    cap: i32,
    out_total: *mut i32,
) -> i32 {
    let q = text_of(unsafe { bytes(query, query_len) });
    let opts = unsafe { Opts::read(opts) };
    let boosts = (!boosts.is_null() && nboosts > 0)
        .then(|| unsafe { slice::from_raw_parts(boosts, nboosts as usize) });
    let cap = cap.max(0) as usize;
    let (oi, os) = if cap == 0 || out_idx.is_null() || out_score.is_null() {
        (&mut [][..], &mut [][..])
    } else {
        unsafe {
            (
                slice::from_raw_parts_mut(out_idx, cap),
                slice::from_raw_parts_mut(out_score, cap),
            )
        }
    };
    with_ctx(id, |c| {
        let hits = ctx_search(c, &q, opts);
        if !out_total.is_null() {
            unsafe { *out_total = hits.len() as i32 };
        }
        write_out(&hits, boosts, oi, os) as i32
    })
    .unwrap_or_else(|| {
        if !out_total.is_null() {
            unsafe { *out_total = 0 };
        }
        0
    })
}

/// Drop a context and its memory. An unknown handle is ignored.
#[unsafe(no_mangle)]
pub extern "C" fn lvimfuzzy_ctx_drop(id: i32) {
    REGISTRY.with(|r| {
        r.borrow_mut().map.remove(&id);
    });
}

// ─── positions ──────────────────────────────────────────────────────────────

/// The matched characters of `text` (0-based character indices, ascending), or
/// None when it does not match. `all` = also every other occurrence of each
/// term (what a list highlights), else only the scored alignment.
pub fn match_chars(text: &str, query: &str, opts: Opts, all: bool) -> Option<Vec<usize>> {
    let pat = Pattern::parse(query, &opts.config());
    if pat.is_empty() {
        return None;
    }
    if all {
        pat.highlights(text)
    } else {
        pat.positions(text).map(|(_, p)| p)
    }
}

/// Character indices → 1-based byte columns, every byte of each character.
pub fn byte_columns(text: &str, chars: &[usize]) -> Vec<i32> {
    let mut out = Vec::with_capacity(chars.len());
    let mut want = chars.iter().peekable();
    for (ci, (bi, ch)) in text.char_indices().enumerate() {
        let Some(&&w) = want.peek() else { break };
        if w == ci {
            want.next();
            for k in 0..ch.len_utf8() {
                out.push((bi + k + 1) as i32);
            }
        }
    }
    out
}

/// # Safety
/// Pointers valid for their lengths; `opts` may be null.
#[allow(clippy::too_many_arguments)]
unsafe fn positions_into(
    text: *const c_char,
    text_len: i32,
    query: *const c_char,
    query_len: i32,
    opts: *const Opts,
    out: *mut i32,
    max: i32,
    all: bool,
    bytes_out: bool,
) -> i32 {
    let t = text_of(unsafe { bytes(text, text_len) });
    let q = text_of(unsafe { bytes(query, query_len) });
    let opts = unsafe { Opts::read(opts) };
    let Some(chars) = match_chars(&t, &q, opts, all) else {
        return -1;
    };
    let vals: Vec<i32> = if bytes_out {
        byte_columns(&t, &chars)
    } else {
        chars.iter().map(|&c| c as i32).collect()
    };
    let n = vals.len().min(max.max(0) as usize);
    if n > 0 && !out.is_null() {
        unsafe { slice::from_raw_parts_mut(out, n) }.copy_from_slice(&vals[..n]);
    }
    n as i32
}

/// The scored alignment of `query` in `text` as 1-based byte columns (every byte
/// of each matched character). Returns how many were written to `out` (at most
/// `max`), -1 when `text` does not match.
///
/// # Safety
/// Pointers valid for their lengths; `opts` may be null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lvimfuzzy_positions(
    text: *const c_char,
    text_len: i32,
    query: *const c_char,
    query_len: i32,
    opts: *const Opts,
    out: *mut i32,
    max: i32,
) -> i32 {
    unsafe {
        positions_into(
            text, text_len, query, query_len, opts, out, max, false, true,
        )
    }
}

/// What a list highlights in a matching `text`: the alignment plus every other
/// occurrence of each term, as 0-based character indices. Returns the count
/// written (at most `max`), -1 when `text` does not match.
///
/// # Safety
/// Pointers valid for their lengths; `opts` may be null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lvimfuzzy_highlights(
    text: *const c_char,
    text_len: i32,
    query: *const c_char,
    query_len: i32,
    opts: *const Opts,
    out: *mut i32,
    max: i32,
) -> i32 {
    unsafe {
        positions_into(
            text, text_len, query, query_len, opts, out, max, true, false,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_of(items: &[&str], skip_empty: bool) -> i32 {
        let id = lvimfuzzy_ctx_new(skip_empty as i32);
        let mut data = Vec::new();
        let mut offs = vec![0i32];
        for t in items {
            data.extend_from_slice(t.as_bytes());
            offs.push(data.len() as i32);
        }
        unsafe { lvimfuzzy_ctx_push(id, data.as_ptr() as _, offs.as_ptr(), items.len() as i32) };
        id
    }

    fn rank(
        id: i32,
        q: &str,
        opts: Option<Opts>,
        boosts: Option<&[i32]>,
    ) -> (Vec<(i32, i32)>, i32) {
        let mut oi = vec![0i32; 1000];
        let mut os = vec![0i32; 1000];
        let mut total = 0;
        let o = opts.unwrap_or_default();
        let (bp, bn) = boosts.map_or((std::ptr::null(), 0), |b| (b.as_ptr(), b.len() as i32));
        let n = unsafe {
            lvimfuzzy_ctx_match(
                id,
                q.as_ptr() as _,
                q.len() as i32,
                &o,
                bp,
                bn,
                oi.as_mut_ptr(),
                os.as_mut_ptr(),
                1000,
                &mut total,
            )
        };
        ((0..n as usize).map(|k| (oi[k], os[k])).collect(), total)
    }

    #[test]
    fn abi_is_seven() {
        assert_eq!(lvimfuzzy_abi(), 7);
    }

    #[test]
    fn push_keeps_empty_candidates_parallel() {
        let id = ctx_of(&["", "abc", "", "xbc"], false);
        assert_eq!(lvimfuzzy_ctx_count(id), 4);
        let (hits, total) = rank(id, "bc", None, None);
        assert_eq!(total, 2);
        assert_eq!(hits.iter().map(|h| h.0).collect::<Vec<_>>(), vec![1, 3]);
        lvimfuzzy_ctx_drop(id);
    }

    #[test]
    fn a_stream_is_the_same_set_however_it_is_cut() {
        let text = b"src/main.rs\nlua/init.lua\n\nREADME.md\nsrc/lib.rs";
        let whole = lvimfuzzy_ctx_new(1);
        unsafe { lvimfuzzy_ctx_append_blob(whole, text.as_ptr() as _, text.len() as i32) };
        lvimfuzzy_ctx_flush(whole);
        let cut = lvimfuzzy_ctx_new(1);
        for ch in text.chunks(3) {
            unsafe { lvimfuzzy_ctx_append_blob(cut, ch.as_ptr() as _, ch.len() as i32) };
        }
        lvimfuzzy_ctx_flush(cut);
        assert_eq!(lvimfuzzy_ctx_count(whole), 4, "the empty line is skipped");
        assert_eq!(rank(whole, "src", None, None), rank(cut, "src", None, None));
        let mut len = 0;
        let p = unsafe { lvimfuzzy_ctx_text(cut, 4, &mut len) };
        let t = unsafe { slice::from_raw_parts(p as *const u8, len as usize) };
        assert_eq!(t, b"src/lib.rs");
        assert!(unsafe { lvimfuzzy_ctx_text(cut, 5, &mut len) }.is_null());
        lvimfuzzy_ctx_drop(whole);
        lvimfuzzy_ctx_drop(cut);
    }

    #[test]
    fn narrowing_gives_the_fresh_result() {
        let items: Vec<String> = (0..5000)
            .map(|i| format!("dir{}/file_{}.lua", i % 37, i))
            .collect();
        let refs: Vec<&str> = items.iter().map(|s| s.as_str()).collect();
        let typed = ctx_of(&refs, false);
        for q in ["f", "fi", "fil", "file", "file_1", "file_12"] {
            let fresh = ctx_of(&refs, false);
            assert_eq!(
                rank(typed, q, None, None),
                rank(fresh, q, None, None),
                "query {q}"
            );
            lvimfuzzy_ctx_drop(fresh);
        }
        lvimfuzzy_ctx_drop(typed);
    }

    #[test]
    fn extended_syntax_and_and() {
        let id = ctx_of(&["lua/test.lua", "lua/init.lua", "src/init.rs"], false);
        let names = |q: &str| {
            rank(id, q, None, None)
                .0
                .iter()
                .map(|h| h.0)
                .collect::<Vec<_>>()
        };
        assert_eq!(names("lua !test"), vec![1]);
        assert_eq!(names("init lua"), vec![1]);
        assert_eq!(names(".rs$"), vec![2]);
        let plain = Opts {
            extended: 0,
            ..Opts::default()
        };
        assert!(
            rank(id, "lua !test", Some(plain), None).0.is_empty(),
            "no syntax: one literal needle"
        );
        lvimfuzzy_ctx_drop(id);
    }

    #[test]
    fn boosts_reorder_and_keep_ties_stable() {
        let id = ctx_of(&["abc", "abd", "abe"], false);
        let (plain, _) = rank(id, "ab", None, None);
        assert_eq!(plain.iter().map(|h| h.0).collect::<Vec<_>>(), vec![0, 1, 2]);
        let (b, _) = rank(id, "ab", None, Some(&[0, 0, 5]));
        assert_eq!(b.iter().map(|h| h.0).collect::<Vec<_>>(), vec![2, 0, 1]);
        assert_eq!(b[0].1, plain[2].1 + 5);
        lvimfuzzy_ctx_drop(id);
    }

    #[test]
    fn grep_scopes() {
        let id = ctx_of(
            &["lua/health.lua:3:local x", "lua/init.lua:9:health()"],
            false,
        );
        let names = |s: i32| {
            rank(
                id,
                "health",
                Some(Opts {
                    scope: s,
                    ..Opts::default()
                }),
                None,
            )
            .0
            .iter()
            .map(|h| h.0)
            .collect::<Vec<_>>()
        };
        assert_eq!(names(1), vec![1], "line");
        assert_eq!(names(2), vec![0], "path");
        lvimfuzzy_ctx_drop(id);
    }

    #[test]
    fn positions_are_byte_columns_of_whole_characters() {
        let chars = match_chars("Пиши", "пи", Opts::default(), false).unwrap();
        assert_eq!(chars, vec![0, 1]);
        assert_eq!(byte_columns("Пиши", &chars), vec![1, 2, 3, 4]);
        assert_eq!(byte_columns("FooBar", &[0, 3]), vec![1, 4]);
        let mut out = [0i32; 16];
        let (t, q) = ("abc", "zz");
        let n = unsafe {
            lvimfuzzy_positions(
                t.as_ptr() as _,
                3,
                q.as_ptr() as _,
                2,
                std::ptr::null(),
                out.as_mut_ptr(),
                16,
            )
        };
        assert_eq!(n, -1);
    }

    #[test]
    fn highlights_mark_every_occurrence() {
        let h = match_chars("ab ab", "ab", Opts::default(), true).unwrap();
        assert_eq!(h, vec![0, 1, 3, 4]);
    }
}
