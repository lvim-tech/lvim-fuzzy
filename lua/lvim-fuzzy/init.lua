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
]])

---@type ffi.namespace*? the loaded native library, or nil when unavailable
local lib
---@type string? absolute path the library was loaded from
local lib_path
---@type integer? the loaded library's reported ABI
local lib_abi
---@type boolean whether the loaded library supports incremental append (ABI ≥ 2)
local has_append = false
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
---@return { index: integer, score: integer }[] results
---@return integer count
function M.match(query, ctx)
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
    local cnt = lib.lvimfuzzy_match_prepared(query, #query, oi, os, cap)
    local out = {}
    for k = 1, cnt do
        out[k] = { index = oi[k - 1] + 1, score = os[k - 1] }
    end
    return out, cnt
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
    ctx.candidates = {}
    ctx.n = 0
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
    return ABI_APPEND
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
    active_token = nil
end

return M
