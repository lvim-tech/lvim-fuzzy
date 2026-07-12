-- lvim-fuzzy: the shared native fuzzy-matching ENGINE for the lvim-tech set
-- (the role blink.lib plays for blink.cmp) — used by lvim-cmp and lvim-picker so
-- they rank one identical way. The hot path is a Rust cdylib (native/) loaded
-- IN-PROCESS via LuaJIT FFI: an affine-gap local-alignment scorer with
-- inter-candidate AVX2 SIMD, a signature prefilter, and a deterministic total
-- order. When the .so is absent (no Rust toolchain) it transparently falls back
-- to a BYTE-IDENTICAL pure-Lua matcher (fallback.lua), so results never differ.
--
-- The boundary is COARSE-GRAINED: the candidate set is uploaded ONCE per
-- completion/pick (prepare), then each keystroke is a single FFI call carrying
-- only the query and returning sorted indices + scores — never a call per
-- candidate. This amortises the cgo/FFI cost and bounds GC pressure.
--
-- Public API:
--   • require("lvim-fuzzy").prepare(candidates) → ctx   (build once per set)
--   • require("lvim-fuzzy").match(query, ctx)   → results, count  (per keystroke)
--   • require("lvim-fuzzy").filter(candidates, query) → results, count  (one-off)
--   • require("lvim-fuzzy").positions(candidate, query) → cols?  (visible rows only)
--   • require("lvim-fuzzy").backend()  → "native-avx2" | "native-scalar" | "lua"
--
---@module "lvim-fuzzy"

local ffi = require("ffi")
local config = require("lvim-fuzzy.config")
local fallback = require("lvim-fuzzy.fallback")

local M = {}

-- ABI contract. The BASE contract (prepare / match_prepared / match / avx2) is
-- ABI ≥ 1; incremental `append` + `ctx_free` arrived in ABI 2. The loader accepts
-- any library reporting ABI ≥ ABI_MIN (so an older ABI-1 .so still loads and works)
-- and exposes the ABI-2 features only when the library reports ≥ ABI_APPEND.
local ABI_MIN = 1
local ABI_APPEND = 2
local ABI_CHUNK = 3
local ABI_PAR = 4
local ABI_BLOB = 5
local ABI_BOOST = 6

ffi.cdef([[
int lvimfuzzy_abi(void);
int lvimfuzzy_avx2(void);
int lvimfuzzy_prepare(const char *hay, const int *offs, int n);
int lvimfuzzy_append(const char *hay, const int *offs, int n);
void lvimfuzzy_ctx_free(void);
int lvimfuzzy_match_prepared(const char *needle, int needle_len,
                             int *out_idx, int *out_score, int max_out);
int lvimfuzzy_match(const char *hay, const int *offs, int n,
                    const char *needle, int needle_len,
                    int *out_idx, int *out_score, int max_out);
int lvimfuzzy_match_chunk_begin(const char *needle, int needle_len);
int lvimfuzzy_match_chunk_step(int budget);
int lvimfuzzy_match_chunk_finish(int *out_idx, int *out_score, int max_out);
int lvimfuzzy_match_prepared_par(const char *needle, int needle_len,
                                 int *out_idx, int *out_score, int max_out);
int lvimfuzzy_match_prepared_boost(const char *needle, int needle_len,
                                   const int *boosts, int boosts_len,
                                   int *out_idx, int *out_score, int max_out);
int lvimfuzzy_ctx_new(void);
int lvimfuzzy_ctx_append_blob(int id, const char *data, int len);
int lvimfuzzy_ctx_flush(int id);
int lvimfuzzy_ctx_count(int id);
const char *lvimfuzzy_ctx_text(int id, int idx, int *out_len);
int lvimfuzzy_ctx_match_par(int id, const char *needle, int needle_len,
                            int *out_idx, int *out_score, int max_out, int *out_total);
void lvimfuzzy_ctx_drop(int id);
]])

---@type ffi.namespace*? the loaded native library, or nil when unavailable
local lib
---@type string? absolute path the library was loaded from
local lib_path
---@type integer? the loaded library's reported ABI
local lib_abi
---@type boolean whether the loaded library supports incremental append (ABI ≥ 2)
local has_append = false
---@type boolean whether the loaded library supports the cooperative chunked match (ABI ≥ 3)
local has_chunk = false
---@type boolean whether the loaded library supports the parallel prepared match (ABI ≥ 4)
local has_par = false
---@type boolean whether the loaded library supports handle-addressed blob-ingest contexts (ABI ≥ 5)
local has_blob = false
---@type boolean whether the loaded library supports the boosted prepared match (ABI ≥ 6)
local has_boost = false
---@type boolean whether load_native() has run
local probed = false
---@type boolean one-shot guard for the fallback notification
local warned = false

-- ─── native library discovery + load ─────────────────────────────────────────

--- Candidate paths for the shared library, in probe order.
---@return string[]
local function candidate_paths()
    local paths = {}
    if config.library_path and config.library_path ~= "" then
        paths[#paths + 1] = config.library_path
    end
    local env = vim.env.LVIM_FUZZY_LIB
    if env and env ~= "" then
        paths[#paths + 1] = env
    end
    -- this file is <root>/lua/lvim-fuzzy/init.lua → strip to the plugin root
    local src = vim.fs.normalize(debug.getinfo(1, "S").source:sub(2))
    local root = src:gsub("/lua/lvim%-fuzzy/init%.lua$", "")
    for _, ext in ipairs({ "so", "dylib", "dll" }) do
        paths[#paths + 1] = root .. "/native/build/liblvimfuzzy." .. ext
        paths[#paths + 1] = root .. "/native/target/release/liblvimfuzzy." .. ext
    end
    return paths
end

--- Try to load a valid native library (exists, loads, ABI matches). Sets `lib`
--- and `lib_path` on success. Idempotent — only probes once.
---@return ffi.namespace*?
local function load_native()
    if probed then
        return lib
    end
    probed = true
    if config.backend == "lua" then
        return nil
    end
    for _, p in ipairs(candidate_paths()) do
        if vim.uv.fs_stat(p) then
            local ok, loaded = pcall(ffi.load, p)
            if ok then
                local abi_ok, abi = pcall(function()
                    return loaded.lvimfuzzy_abi()
                end)
                if abi_ok and abi >= ABI_MIN then
                    lib = loaded
                    lib_path = p
                    lib_abi = abi
                    has_append = abi >= ABI_APPEND
                    has_chunk = abi >= ABI_CHUNK
                    has_par = abi >= ABI_PAR
                    has_blob = abi >= ABI_BLOB
                    has_boost = abi >= ABI_BOOST
                    return lib
                end
            end
        end
    end
    return nil
end

-- ─── reused FFI output buffers (grown, never shrunk) ─────────────────────────
---@type ffi.cdata*?
local out_idx
---@type ffi.cdata*?
local out_score
---@type integer
local out_cap = 0

--- Ensure the output buffers hold at least `n` int slots; return them (non-nil).
---@param n integer
---@return ffi.cdata* out_idx
---@return ffi.cdata* out_score
local function ensure_out(n)
    if n > out_cap or not out_idx or not out_score then
        out_cap = n
        out_idx = ffi.new("int[?]", n)
        out_score = ffi.new("int[?]", n)
    end
    return out_idx, out_score
end

-- The native prepared context is a single active slot on the Rust side. Each ctx
-- gets a token; a match re-uploads its set only if a DIFFERENT ctx was prepared
-- since (so two consumers — lvim-cmp and lvim-picker — interleave correctly).
---@type integer
local token_seq = 0
---@type integer?
local active_token

--- Effective result cap for a set of `n` candidates.
---@param n integer
---@return integer
local function result_cap(n)
    local mr = config.max_results or 0
    if mr > 0 then
        return math.min(n, mr)
    end
    return n
end

--- Notify once (INFO) that the Lua fallback is in use, if configured.
local function maybe_warn_fallback()
    if config.warn_on_fallback and not warned then
        warned = true
        vim.schedule(function()
            vim.notify(
                "lvim-fuzzy: native library not found — using the pure-Lua matcher "
                    .. "(build it with `sh native/build.sh`, needs a Rust toolchain).",
                vim.log.levels.INFO
            )
        end)
    end
end

--- Build the concatenated blob + (n+1) offsets for `count` candidates starting at
--- `from` (1-based) in `candidates` — the shape native prepare/append receive.
---@param candidates string[]
---@param from integer
---@param count integer
---@return string blob, ffi.cdata* offs
local function build_blob(candidates, from, count)
    local parts = {}
    local offs = ffi.new("int[?]", count + 1)
    local pos = 0
    for i = 1, count do
        offs[i - 1] = pos
        local s = candidates[from + i - 1]
        parts[i] = s
        pos = pos + #s
    end
    offs[count] = pos
    return table.concat(parts), offs
end

--- Native-prepare the WHOLE current candidate set of `ctx` and mark it active.
--- Re-derives the blob from ctx.candidates so it is always current (correct even
--- after appends made while another context was active).
---@param ctx LvimFuzzyContext
local function upload_ctx(ctx)
    if not lib then
        return
    end
    local blob, offs = build_blob(ctx.candidates, 1, ctx.n)
    lib.lvimfuzzy_prepare(blob, offs, ctx.n)
    active_token = ctx.token
end

-- ─── public API ──────────────────────────────────────────────────────────────

--- Build a matching context for a candidate set. Do this ONCE per completion /
--- pick; then call `match(query, ctx)` on every keystroke. The set can grow with
--- `append(ctx, more)` without rebuilding what is already there.
---@param candidates string[]  the haystack (searched by their bytes)
---@return LvimFuzzyContext
function M.prepare(candidates)
    ---@class LvimFuzzyContext
    ---@field candidates string[]  the full (growable) candidate list
    ---@field n integer            number of candidates
    ---@field native boolean       whether this ctx uses the native backend
    ---@field token integer        identity for the single-active-slot re-upload
    local ctx = { candidates = candidates, n = #candidates, native = false, token = 0 }

    if load_native() and lib then
        token_seq = token_seq + 1
        ctx.native = true
        ctx.token = token_seq
        upload_ctx(ctx) -- native prepare COPIES the blob; nothing kept Lua-side
    else
        maybe_warn_fallback()
    end
    return ctx
end

--- Append NEW candidates to an existing context, extending it INCREMENTALLY:
--- only the new candidates' data is built (native side: their offsets/signatures/
--- lengths are appended in place — the existing pool is never re-scanned). A
--- subsequent `match` ranks the whole grown set. Result is byte-identical to a
--- `prepare(old ++ new)`.
---@param ctx LvimFuzzyContext
---@param candidates string[]  the new candidates to add
---@return integer count  the new total candidate count
function M.append(ctx, candidates)
    local add = #candidates
    if add == 0 then
        return ctx.n
    end
    -- extend the Lua-side list (used by the fallback and by any whole re-upload)
    local new_n = fallback.append(ctx.candidates, candidates)
    local first_new = ctx.n + 1
    ctx.n = new_n

    if ctx.native and lib then
        if has_append and active_token == ctx.token then
            -- fast path: build ONLY the new candidates and extend the native SoA
            local blob, offs = build_blob(ctx.candidates, first_new, add)
            lib.lvimfuzzy_append(blob, offs, add)
            -- native now holds the grown set; ctx stays the active slot
        elseif active_token == ctx.token then
            -- library too old for incremental append → force a whole re-upload next match
            active_token = nil
        end
        -- (ctx not active → active_token already ≠ ctx.token → match() re-uploads whole)
    end
    return ctx.n
end

--- Rank `ctx`'s candidates against `query`. Returns a list of
--- `{ index, score }` (1-based `index` into the candidate array) in ranked
--- order, plus the count. Deterministic; identical across backends.
---@param query string
---@param ctx LvimFuzzyContext
---@param boosts integer[]?  per-candidate score offsets (1-based, parallel to the prepared
---   candidate set): `boosts[i]` is ADDED to candidate i's match score before ranking, so a
---   caller (lvim-cmp) can fold exact-prefix / proximity / frecency signals into the fuzzy
---   order. Applied only on the native ABI ≥ 6 library; ignored on the pure-Lua fallback.
---@return { index: integer, score: integer }[] results
---@return integer count
function M.match(query, ctx, boosts)
    if not ctx.native or not lib then
        return fallback.match(ctx.candidates, query, config.max_results or 0)
    end
    -- re-upload the whole (current) set if another ctx became active since this one
    -- was prepared, or after an append made while it was not the active slot
    if active_token ~= ctx.token then
        upload_ctx(ctx)
    end
    local cap = result_cap(ctx.n)
    local oi, os = ensure_out(math.max(cap, 1))
    local cnt
    if boosts and has_boost then
        -- copy the Lua boost array into a native int32 buffer (parallel to the candidate set)
        local nb = ctx.n
        local bbuf = ffi.new("int32_t[?]", math.max(nb, 1))
        for i = 1, nb do
            bbuf[i - 1] = boosts[i] or 0
        end
        cnt = lib.lvimfuzzy_match_prepared_boost(query, #query, bbuf, nb, oi, os, cap)
    else
        cnt = lib.lvimfuzzy_match_prepared(query, #query, oi, os, cap)
    end
    local out = {}
    for k = 1, cnt do
        out[k] = { index = oi[k - 1] + 1, score = os[k - 1] }
    end
    return out, cnt
end

---@type integer  bumped on every match_async; an in-flight chunked match whose gen is superseded aborts
local async_gen = 0
---@type string?  query of the in-flight async match. An identical re-request COALESCES onto it instead of
--- restarting it: a stream-refresh (or a repeated keystroke) firing the SAME query while a match runs must not
--- keep superseding it, or over a huge pool — where one match is ~250 ms — nothing would ever reach completion.
--- A DIFFERENT query still supersedes immediately. Cleared when the match it belongs to finishes / its ctx dies.
local async_query
---@type LvimFuzzyContext?  ctx of the in-flight async match (the coalesce key is the (query, ctx) pair, so the
--- same query against a freshly prepared context still runs).
local async_ctx
---@type fun(results: any[], count: integer)?  delivery callback of the in-flight match. A COALESCED identical
--- request swaps in its OWN callback so completion delivers to the NEWEST caller — the caller (the picker)
--- bumps a generation on every request and drops a result whose generation is stale, so delivering to the
--- match's ORIGINAL (now-superseded) callback would be silently discarded. Redirecting keeps the guard happy.
local async_on_result

--- Rank `ctx` against `query` ASYNCHRONOUSLY: the match is driven in `config.chunk_budget`-sized slices across
--- event-loop ticks (the ABI-3 chunked path), so a match over a HUGE pool (hundreds of thousands / millions of
--- candidates, where one match is tens–hundreds of ms) never blocks the editor in a single FFI call — the UI
--- redraws and input is handled between slices. `on_result(results, count)` fires on a later tick. A newer
--- `match_async` SUPERSEDES an in-flight one (the older result is dropped). When the library is ABI < 3 or the
--- pure-Lua fallback is active, it does the plain synchronous match, delivered on the next tick to keep the
--- async shape. Byte-identical results to `match` (proven: chunked ≡ whole).
---@param query string
---@param ctx LvimFuzzyContext
---@param on_result fun(results: { index: integer, score: integer }[], count: integer)
function M.match_async(query, ctx, on_result)
    -- COALESCE an identical in-flight request: the same (query, ctx) is already matching, so let it finish and
    -- deliver rather than restarting it (repeated restarts from the stream-refresh firing the SAME query would
    -- keep superseding it). A different query / ctx falls through and supersedes.
    if query == async_query and ctx == async_ctx then
        async_on_result = on_result -- redirect the running match's delivery to this newest caller (see field)
        return
    end
    async_gen = async_gen + 1
    local mygen = async_gen
    async_query, async_ctx, async_on_result = query, ctx, on_result
    --- Release the in-flight marker, but only if THIS match is still the current one (a newer query that
    --- superseded us already owns the marker — leave it for that match).
    local function clear_inflight()
        if mygen == async_gen then
            async_query, async_ctx, async_on_result = nil, nil, nil
        end
    end

    -- PARALLEL (ABI 4): the scan is fanned out across CPU cores, so even a match over MILLIONS of candidates is
    -- tens of ms — fast enough to run in ONE shot per keystroke and still keep up with typing. Run it on a
    -- SCHEDULED tick so it never blocks the keystroke's own input handling, then deliver. No cooperative
    -- chunking needed (that path — below — exists only for an older .so that predates the parallel entry).
    if ctx.native and lib and has_par then
        vim.schedule(function()
            if mygen ~= async_gen then
                return
            end
            -- Rare: another ctx became active since this one (or a not-active append) → one whole re-prepare.
            if active_token ~= ctx.token then
                upload_ctx(ctx)
            end
            local cap = result_cap(ctx.n)
            local oi, os = ensure_out(math.max(cap, 1))
            local cnt = lib.lvimfuzzy_match_prepared_par(query, #query, oi, os, cap)
            local out = {}
            for k = 1, cnt do
                out[k] = { index = oi[k - 1] + 1, score = os[k - 1] }
            end
            local deliver = async_on_result or on_result -- newest coalesced caller, if any
            clear_inflight()
            deliver(out, cnt)
        end)
        return
    end

    -- ABI < 3 or Lua fallback → plain sync match, delivered next tick (same async contract as the paths above)
    if not (ctx.native and lib and has_chunk) then
        vim.schedule(function()
            if mygen ~= async_gen then
                return
            end
            local res, cnt = M.match(query, ctx)
            local deliver = async_on_result or on_result
            clear_inflight()
            deliver(res, cnt)
        end)
        return
    end

    -- ABI 3 cooperative CHUNKED match (a .so that has the chunked entry but not the parallel one): drive the
    -- match in budget-sized slices across event-loop ticks so a huge single-threaded match never blocks.
    if active_token ~= ctx.token then
        upload_ctx(ctx)
    end
    local n = lib.lvimfuzzy_match_chunk_begin(query, #query)
    local budget = config.chunk_budget or 65536
    -- Yield between slices with a libuv TIMER (vim.defer_fn), NOT vim.schedule: scheduled callbacks are drained
    -- in one loop pass (no redraw / input poll between them), whereas a 0-ms timer fires on the NEXT loop
    -- iteration — AFTER the poll phase handles input and the screen redraws — actually keeping the editor live.
    local function step()
        -- Abort a STALE slice: a newer query superseded this one (async_gen), OR the active prepared context
        -- changed out from under us (another ctx prepared, or this finder freed its context → active_token=nil).
        if mygen ~= async_gen or active_token ~= ctx.token then
            clear_inflight() -- if the ctx died under the current match, free the marker so later matches run
            return
        end
        if lib.lvimfuzzy_match_chunk_step(budget) < n then
            vim.defer_fn(step, 0)
            return
        end
        local cap = result_cap(ctx.n)
        local oi, os = ensure_out(math.max(cap, 1))
        local cnt = lib.lvimfuzzy_match_chunk_finish(oi, os, cap)
        local out = {}
        for k = 1, cnt do
            out[k] = { index = oi[k - 1] + 1, score = os[k - 1] }
        end
        local deliver = async_on_result or on_result
        clear_inflight()
        if mygen == async_gen then
            deliver(out, cnt)
        end
    end
    vim.defer_fn(step, 0)
end

--- One-off stateless rank (build + match in a single call) — convenience for
--- callers that do not keep a context. Uses the native STATELESS entry point so
--- it never disturbs an active prepared context.
---@param candidates string[]
---@param query string
---@return { index: integer, score: integer }[] results
---@return integer count
function M.filter(candidates, query)
    local n = #candidates
    if not load_native() or not lib then
        maybe_warn_fallback()
        return fallback.match(candidates, query, config.max_results or 0)
    end
    local blob = table.concat(candidates)
    local offs = ffi.new("int[?]", n + 1)
    local pos = 0
    for i = 1, n do
        offs[i - 1] = pos
        pos = pos + #candidates[i]
    end
    offs[n] = pos
    local cap = result_cap(n)
    local oi, os = ensure_out(math.max(cap, 1))
    local cnt = lib.lvimfuzzy_match(blob, offs, n, query, #query, oi, os, cap)
    local out = {}
    for k = 1, cnt do
        out[k] = { index = oi[k - 1] + 1, score = os[k - 1] }
    end
    return out, cnt
end

--- Matched-char BYTE columns (1-based, ascending) of `query` inside `candidate` —
--- the exact alignment the scorer credits, so highlighted chars always agree with the
--- ranking. Computed in Lua on EVERY backend (both share the constants/recurrence, so
--- the result is identical either way): a menu needs positions only for its ~12
--- VISIBLE rows, which is far below FFI-crossing weight. Returns nil on a non-match.
---@param candidate string
---@param query string
---@return integer[]?
function M.positions(candidate, query)
    return fallback.positions(candidate, query)
end

--- Release a context. On the native backend, if `ctx` is the active slot this
--- frees the native SoA memory eagerly (a large pool can be reclaimed instead of
--- lingering until the next `prepare`). The Lua-side lists are dropped by GC once
--- the caller lets `ctx` go. Optional — not calling it is harmless.
---@param ctx LvimFuzzyContext
function M.free(ctx)
    if ctx.native and lib and has_append and active_token == ctx.token then
        lib.lvimfuzzy_ctx_free()
        active_token = nil
    end
    if async_ctx == ctx then -- a match on this ctx was in flight → drop the coalesce marker so it can't linger
        async_query, async_ctx, async_on_result = nil, nil, nil
    end
    ctx.candidates = {}
    ctx.n = 0
end

-- ─── handle-addressed BLOB-ingest contexts (ABI 5) ───────────────────────────
-- For a STREAMING finder over a huge tree (millions of paths): instead of materialising every candidate as a
-- Lua string + table (a multi-second load of LuaJIT string-intern-table doublings + GC over the pool), the
-- producer's raw stdout BYTES are handed straight to a native context, which splits on '\n' and stores each
-- line as a candidate. The consumer keeps only the visible / top-K rows, materialised on demand via
-- `blob_text`. Each context is a HANDLE (an id in a native registry), never the single active prepare/match
-- slot — so a huge pool can never be clobbered by another consumer preparing in between (and, unlike that
-- slot, it needs no Lua-side re-upload copy). Requires ABI ≥ 5; callers gate on `M.has_blob()` and keep the
-- current per-string path for an older .so.
---@class LvimFuzzyBlob
---@field id integer  the native context handle
---@field n integer   the current candidate count

---@type ffi.cdata* reusable single-int out buffer for blob_text's length
local len_buf = ffi.new("int[1]")
---@type ffi.cdata* reusable single-int out buffer for blob_match's TRUE total match count (before the cap)
local total_buf = ffi.new("int[1]")

--- Create a new blob-ingest context, or nil when the native library is absent / predates ABI 5 (the caller
--- then keeps the current per-string streaming path).
---@return LvimFuzzyBlob?
function M.blob_new()
    if not (load_native() and lib and has_blob) then
        return nil
    end
    return { id = lib.lvimfuzzy_ctx_new(), n = 0 }
end

--- Ingest a raw byte chunk (a producer's stdout, arbitrary boundaries — the native side carries a straddling
--- line to the next chunk). Returns the new candidate count. ZERO Lua per-row work.
---@param blob LvimFuzzyBlob
---@param data string  raw bytes
---@return integer count
function M.blob_append(blob, data)
    if not lib then
        return blob.n
    end
    blob.n = lib.lvimfuzzy_ctx_append_blob(blob.id, data, #data)
    return blob.n
end

--- Flush a trailing partial line (call once at end of stream). Returns the candidate count.
---@param blob LvimFuzzyBlob
---@return integer count
function M.blob_flush(blob)
    if not lib then
        return blob.n
    end
    blob.n = lib.lvimfuzzy_ctx_flush(blob.id)
    return blob.n
end

--- The current candidate count of `blob`.
---@param blob LvimFuzzyBlob
---@return integer
function M.blob_count(blob)
    if not lib then
        return blob.n
    end
    return lib.lvimfuzzy_ctx_count(blob.id)
end

--- The text of candidate `idx` (1-based). Copies the bytes out of the native blob immediately (before any
--- further ingest could move them). Nil for an out-of-range index.
---@param blob LvimFuzzyBlob
---@param idx integer  1-based candidate index
---@return string?
function M.blob_text(blob, idx)
    if not lib then
        return nil
    end
    local p = lib.lvimfuzzy_ctx_text(blob.id, idx, len_buf)
    if p == nil then
        return nil
    end
    return ffi.string(p, len_buf[0])
end

---@type integer  bumped on every blob_match; a superseded scheduled match aborts (cheap no-op)
local blob_async_gen = 0

--- Rank `blob` against `query` ASYNCHRONOUSLY (parallel scan, delivered on a scheduled tick so it never blocks
--- the keystroke that triggered it). `on_result(results, count, total)` — `results[k] = { index, score }`
--- (1-based index into the candidate set), `count` = rows returned (≤ max_results), `total` = the TRUE number
--- of candidates that matched (before the cap; the whole pool for an empty query) so a caller can show
--- fzf-style `matched/total`. Same shape as `match_async` plus the total. A newer `blob_match` supersedes an
--- in-flight one. Positions for highlighting are the CALLER's (visible rows only), exactly like `match_async`.
---@param blob LvimFuzzyBlob
---@param query string
---@param on_result fun(results: { index: integer, score: integer }[], count: integer, total: integer)
function M.blob_match(blob, query, on_result)
    blob_async_gen = blob_async_gen + 1
    local mygen = blob_async_gen
    if not (load_native() and lib and has_blob) then
        vim.schedule(function()
            if mygen == blob_async_gen then
                on_result({}, 0, 0)
            end
        end)
        return
    end
    vim.schedule(function()
        if mygen ~= blob_async_gen then
            return
        end
        local cap = result_cap(blob.n)
        local oi, os = ensure_out(math.max(cap, 1))
        local cnt = lib.lvimfuzzy_ctx_match_par(blob.id, query, #query, oi, os, cap, total_buf)
        local out = {}
        for k = 1, cnt do
            out[k] = { index = oi[k - 1] + 1, score = os[k - 1] }
        end
        on_result(out, cnt, total_buf[0])
    end)
end

--- Drop a blob context, releasing its native memory. Optional but recommended on finder close (a huge pool is
--- reclaimed eagerly). Safe to call more than once.
---@param blob LvimFuzzyBlob?
function M.blob_free(blob)
    if blob and lib and has_blob then
        lib.lvimfuzzy_ctx_drop(blob.id)
        blob.n = 0
    end
end

-- ─── introspection (used by health.lua and consumers) ────────────────────────

--- Which backend is active: "native-avx2", "native-scalar", or "lua".
---@return string
function M.backend()
    if load_native() and lib then
        return (lib.lvimfuzzy_avx2() == 1) and "native-avx2" or "native-scalar"
    end
    return "lua"
end

--- True when the native library is loaded and ABI-compatible.
---@return boolean
function M.native_loaded()
    return load_native() ~= nil and lib ~= nil
end

--- The path the native library was loaded from, or nil.
---@return string?
function M.library_path()
    load_native()
    return lib_path
end

--- The highest ABI version this Lua understands (the base contract is ABI_MIN).
---@return integer
function M.abi()
    return ABI_BOOST
end

--- The minimum ABI the loader accepts.
---@return integer
function M.abi_min()
    return ABI_MIN
end

--- Whether the loaded native library supports incremental append (ABI ≥ 2).
---@return boolean
function M.has_append()
    load_native()
    return has_append
end

--- Whether the loaded native library supports the cooperative chunked async match (ABI ≥ 3) — i.e. whether
--- `match_async` runs non-blocking (driven in slices) rather than delivering a synchronous match on a tick.
---@return boolean
function M.has_chunk()
    load_native()
    return has_chunk
end

--- Whether the loaded native library supports the PARALLEL prepared match (ABI ≥ 4) — the scan fanned out
--- across CPU cores, so `match_async` runs a huge-pool match in one fast shot instead of the slower chunked path.
---@return boolean
function M.has_par()
    load_native()
    return has_par
end

--- Whether the loaded native library supports handle-addressed BLOB-ingest contexts (ABI ≥ 5) — the streaming
--- path that ingests raw producer bytes with no Lua-side candidate strings.
---@return boolean
function M.has_blob()
    load_native()
    return has_blob
end

--- Whether the loaded native library supports the BOOSTED prepared match (ABI ≥ 6) — a
--- per-candidate score-offset array folded in before ranking (lvim-cmp's exact-prefix /
--- proximity boosts). Falls back to the plain match (boosts ignored) when unsupported.
---@return boolean
function M.has_boost()
    load_native()
    return has_boost
end

--- The native library's reported ABI (nil when not loaded).
---@return integer?
function M.native_abi()
    if load_native() and lib then
        return lib.lvimfuzzy_abi()
    end
    return nil
end

--- Merge user options into the live config (in place).
---@param opts? LvimFuzzyConfig
function M.setup(opts)
    local ok_utils, utils = pcall(require, "lvim-utils.utils")
    if ok_utils and utils.merge then
        utils.merge(config, opts or {})
    elseif opts then
        for k, v in pairs(opts) do
            config[k] = v
        end
    end
    -- allow re-probing if the backend preference or path changed
    probed = false
    lib = nil
    lib_path = nil
    lib_abi = nil
    has_append = false
    has_chunk = false
    has_par = false
    has_blob = false
    has_boost = false
    active_token = nil
end

return M
