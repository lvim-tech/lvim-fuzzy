-- lvim-fuzzy: the fuzzy search engine of the lvim-tech set in Neovim — used by lvim-picker and lvim-cmp.
-- The engine is lvim-fuzzy-core (Rust), the same one lvim-fuzzy-cli runs on, so a query ranks the same
-- in the picker, in completion and in the terminal. It is reached IN-PROCESS through a thin C ABI
-- (native/, a cdylib) loaded with LuaJIT FFI. The library is required: there is no pure-Lua matcher.
--
-- The boundary is COARSE-GRAINED: a candidate set lives in a native CONTEXT (a handle), filled once
-- (`prepare`, `append`) or straight from a process's output (`blob_*`), and every keystroke is ONE call
-- carrying only the query and returning the ranked indices. While the query only grows, the native side
-- searches only the previous matches.
--
-- Queries take fzf's extended syntax by default: spaces separate terms that must all match, `'x` exact,
-- `^x` prefix, `x$` suffix, `!x` negation, `a | b` either; smart case; Latin letters match their accented
-- forms. Every search call takes an optional `opts` table (see `LvimFuzzyOpts`) over `config`.
--
-- Public API:
--   • prepare(candidates, opts?) → ctx · append(ctx, more) · free(ctx)
--   • match(query, ctx, boosts?, opts?) → results, count, total · match_async(query, ctx, cb, opts?)
--   • filter(candidates, query, opts?) → results, count, total            (one-off)
--   • positions(text, query, opts?) → byte columns? · highlights(text, query, opts?) → char indices?
--   • blob_new / blob_append / blob_flush / blob_count / blob_text / blob_match / blob_free
--   • backend() → "native-avx2" | "native-scalar" | "none"
--
---@module "lvim-fuzzy"

local ffi = require("ffi")
local config = require("lvim-fuzzy.config")

local M = {}

--- The one ABI this Lua speaks: 7 = the library over lvim-fuzzy-core. An older library (ABI 1–6, its own
--- matcher) is refused, and :checkhealth says how to rebuild.
local ABI = 7

ffi.cdef([[
typedef struct {
    int scope;
    int case_mode;
    int extended;
    int exact;
    int normalize;
    int scheme;
    int typos;
    int exact_window;
} lvimfuzzy_opts;
int lvimfuzzy_abi(void);
int lvimfuzzy_avx2(void);
int lvimfuzzy_ctx_new(int skip_empty);
int lvimfuzzy_ctx_push(int id, const char *data, const int *offs, int n);
int lvimfuzzy_ctx_append_blob(int id, const char *data, int len);
int lvimfuzzy_ctx_flush(int id);
int lvimfuzzy_ctx_count(int id);
const char *lvimfuzzy_ctx_text(int id, int idx, int *out_len);
int lvimfuzzy_ctx_match(int id, const char *query, int query_len, const lvimfuzzy_opts *opts,
                        const int *boosts, int nboosts,
                        int *out_idx, int *out_score, int cap, int *out_total);
void lvimfuzzy_ctx_drop(int id);
int lvimfuzzy_positions(const char *text, int text_len, const char *query, int query_len,
                        const lvimfuzzy_opts *opts, int *out, int max);
int lvimfuzzy_highlights(const char *text, int text_len, const char *query, int query_len,
                         const lvimfuzzy_opts *opts, int *out, int max);
]])

---@type ffi.namespace*? the loaded native library, or nil when unavailable
local lib
---@type string? absolute path the library was loaded from
local lib_path
---@type string? why no library is loaded (for :checkhealth and the error)
local load_error
---@type boolean whether load_native() has run
local probed = false

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

--- Load the native library (exists, loads, ABI matches). Idempotent — probes once.
---@return ffi.namespace*?
local function load_native()
    if probed then
        return lib
    end
    probed = true
    load_error = "no liblvimfuzzy found (build it: sh native/build.sh — needs cargo)"
    for _, p in ipairs(candidate_paths()) do
        if vim.uv.fs_stat(p) then
            local ok, loaded = pcall(ffi.load, p)
            if not ok then
                load_error = ("%s does not load: %s"):format(p, tostring(loaded))
            else
                local abi_ok, abi = pcall(function()
                    return loaded.lvimfuzzy_abi()
                end)
                if abi_ok and abi == ABI then
                    lib, lib_path, load_error = loaded, p, nil
                    return lib
                end
                load_error = ("%s is ABI %s, this lvim-fuzzy needs ABI %d (rebuild: sh native/build.sh)"):format(
                    p,
                    abi_ok and tostring(abi) or "?",
                    ABI
                )
            end
        end
    end
    return nil
end

--- The library, or a clear error: every search entry needs it.
---@return ffi.namespace*
local function need()
    local l = load_native()
    if not l then
        error("lvim-fuzzy: " .. tostring(load_error) .. " — see :checkhealth lvim-fuzzy", 3)
    end
    return l
end

-- ─── options ─────────────────────────────────────────────────────────────────

---@class LvimFuzzyOpts
---@field scope?        "all"|"line"|"path"|"file"  what a grep result (`path:line:text`) is searched by
---@field case?         "smart"|"ignore"|"respect"
---@field extended?     boolean   fzf's extended syntax (false = the query is one literal needle)
---@field exact?        boolean   every term is an exact substring
---@field normalize?    boolean   Latin letters match their accented forms
---@field scheme?       "default"|"path"|"history"
---@field typos?        integer|"auto"  characters a fuzzy term may miss
---@field exact_window? integer   the widest window scored exactly (0 = always exactly)

local SCOPES = { all = 0, line = 1, path = 2, file = 3 }
local CASES = { smart = 0, ignore = 1, respect = 2 }
local SCHEMES = { default = 0, path = 1, history = 2 }

--- `opts` over `config` as the native struct.
---@param opts LvimFuzzyOpts?
---@return ffi.cdata*
local function copts(opts)
    opts = opts or {}
    local function pick(k)
        local v = opts[k]
        if v == nil then
            v = config[k]
        end
        return v
    end
    local typos = pick("typos")
    local window = pick("exact_window")
    return ffi.new("lvimfuzzy_opts", {
        scope = SCOPES[opts.scope or "all"] or 0,
        case_mode = CASES[pick("case") or "smart"] or 0,
        extended = pick("extended") == false and 0 or 1,
        exact = pick("exact") and 1 or 0,
        normalize = pick("normalize") == false and 0 or 1,
        scheme = SCHEMES[pick("scheme") or "default"] or 0,
        typos = typos == "auto" and -1 or tonumber(typos) or 0,
        exact_window = tonumber(window) or -1,
    })
end

-- ─── reused FFI buffers (grown, never shrunk) ────────────────────────────────
---@type ffi.cdata*?
local out_idx
---@type ffi.cdata*?
local out_score
local out_cap = 0
local total_buf = ffi.new("int[1]")
local len_buf = ffi.new("int[1]")
---@type ffi.cdata*?
local pos_buf
local pos_cap = 0

---@param n integer
---@return ffi.cdata*, ffi.cdata*
local function ensure_out(n)
    if n > out_cap or not out_idx or not out_score then
        out_cap = n
        out_idx = ffi.new("int[?]", n)
        out_score = ffi.new("int[?]", n)
    end
    return out_idx, out_score
end

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

-- ─── contexts ────────────────────────────────────────────────────────────────

---@class LvimFuzzyContext
---@field id integer      the native handle
---@field n integer       the candidate count
---@field native boolean  always true (kept for callers that check it)
---@field opts LvimFuzzyOpts?  the options `match` uses when it is given none
---@field _gc ffi.cdata*  drops the native context when the ctx is collected

--- A new native context with a GC guard, so a ctx a caller forgets to free is still reclaimed.
---@param skip_empty boolean
---@return integer id, ffi.cdata* guard
local function new_handle(skip_empty)
    local l = need()
    local id = l.lvimfuzzy_ctx_new(skip_empty and 1 or 0)
    local guard = ffi.gc(ffi.new("char[1]"), function()
        l.lvimfuzzy_ctx_drop(id)
    end)
    return id, guard
end

--- Push a Lua list into context `id` (one packed copy). Returns the new count.
---@param id integer
---@param list string[]
---@return integer
local function push(id, list)
    local n = #list
    if n == 0 then
        return lib.lvimfuzzy_ctx_count(id)
    end
    local offs = ffi.new("int[?]", n + 1)
    local pos = 0
    for i = 1, n do
        offs[i - 1] = pos
        pos = pos + #list[i]
    end
    offs[n] = pos
    return lib.lvimfuzzy_ctx_push(id, table.concat(list), offs, n)
end

--- Rank context `id`; results as `{ index (1-based), score }`.
---@return { index: integer, score: integer }[] results, integer count, integer total
local function run(id, n, query, opts, boosts)
    local cap = result_cap(n)
    local oi, os = ensure_out(math.max(cap, 1))
    local bbuf, nb = nil, 0
    if boosts then
        nb = n
        bbuf = ffi.new("int[?]", math.max(nb, 1))
        for i = 1, nb do
            bbuf[i - 1] = boosts[i] or 0
        end
    end
    local cnt = lib.lvimfuzzy_ctx_match(id, query, #query, copts(opts), bbuf, nb, oi, os, cap, total_buf)
    local out = {}
    for k = 1, cnt do
        out[k] = { index = oi[k - 1] + 1, score = os[k - 1] }
    end
    return out, cnt, total_buf[0]
end

--- Build a context for a candidate set. Do this ONCE per completion / pick, then `match` per keystroke.
--- Every candidate is kept (an empty one too), so result indices are indices into `candidates`.
---@param candidates string[]
---@param opts LvimFuzzyOpts?  the options `match` / `match_async` use when given none
---@return LvimFuzzyContext
function M.prepare(candidates, opts)
    local id, guard = new_handle(false)
    local ctx = { id = id, n = 0, native = true, opts = opts, _gc = guard }
    ctx.n = push(id, candidates)
    return ctx
end

--- Add candidates to a context (only the new ones cross). Returns the new count.
---@param ctx LvimFuzzyContext
---@param candidates string[]
---@return integer
function M.append(ctx, candidates)
    if #candidates > 0 and ctx._gc then
        ctx.n = push(ctx.id, candidates)
    end
    return ctx.n
end

--- Rank `ctx` against `query`: `{ index, score }` (1-based index) best first, the count returned (≤
--- `config.max_results`) and the total that matched. An empty query is the whole set in source order.
---@param query string
---@param ctx LvimFuzzyContext
---@param boosts integer[]?  per-candidate score offsets, parallel to the set: `boosts[i]` is ADDED to
---   candidate i's score before ranking (lvim-cmp's exact-prefix / proximity signals); ties keep the
---   engine's order
---@param opts LvimFuzzyOpts?
---@return { index: integer, score: integer }[] results
---@return integer count
---@return integer total
function M.match(query, ctx, boosts, opts)
    need()
    if not ctx._gc then
        return {}, 0, 0
    end
    return run(ctx.id, ctx.n, query, opts or ctx.opts, boosts)
end

---@type integer  bumped on every match_async; a superseded scheduled match is dropped
local async_gen = 0
---@type string?, LvimFuzzyContext?, function?  the in-flight request (an identical one coalesces onto it)
local async_query, async_ctx, async_on_result

--- `match` on a scheduled tick, so it never runs inside the keystroke that asked for it. A newer request
--- supersedes an older one; an IDENTICAL (query, ctx) request while one is pending coalesces onto it and
--- takes over its delivery (the caller drops results of a stale generation of its own).
---@param query string
---@param ctx LvimFuzzyContext
---@param on_result fun(results: { index: integer, score: integer }[], count: integer, total: integer)
---@param opts LvimFuzzyOpts?
function M.match_async(query, ctx, on_result, opts)
    if query == async_query and ctx == async_ctx then
        async_on_result = on_result
        return
    end
    async_gen = async_gen + 1
    local mygen = async_gen
    async_query, async_ctx, async_on_result = query, ctx, on_result
    vim.schedule(function()
        if mygen ~= async_gen then
            return
        end
        local deliver = async_on_result or on_result
        async_query, async_ctx, async_on_result = nil, nil, nil
        if not ctx._gc then
            return
        end
        local res, cnt, total = M.match(query, ctx, nil, opts)
        deliver(res, cnt, total)
    end)
end

--- One-off rank without a kept context.
---@param candidates string[]
---@param query string
---@param opts LvimFuzzyOpts?
---@return { index: integer, score: integer }[] results
---@return integer count
---@return integer total
function M.filter(candidates, query, opts)
    local ctx = M.prepare(candidates)
    local res, cnt, total = run(ctx.id, ctx.n, query, opts)
    M.free(ctx)
    return res, cnt, total
end

--- Release a context's native memory now (otherwise the GC does it). Safe to call more than once.
---@param ctx LvimFuzzyContext
function M.free(ctx)
    if ctx and ctx._gc then
        ffi.gc(ctx._gc, nil)
        ctx._gc = nil
        if lib then
            lib.lvimfuzzy_ctx_drop(ctx.id)
        end
    end
    if async_ctx == ctx then
        async_query, async_ctx, async_on_result = nil, nil, nil
    end
    if ctx then
        ctx.n = 0
    end
end

-- ─── positions ───────────────────────────────────────────────────────────────

---@param fn string  the native entry
---@param text string
---@param query string
---@param opts LvimFuzzyOpts?
---@return integer[]?
local function positions_of(fn, text, query, opts)
    if query == "" or text == "" then
        return nil
    end
    local l = need()
    local cap = #text
    if cap > pos_cap or not pos_buf then
        pos_cap = math.max(cap, 256)
        pos_buf = ffi.new("int[?]", pos_cap)
    end
    local n = l[fn](text, #text, query, #query, copts(opts), pos_buf, pos_cap)
    if n < 0 then
        return nil
    end
    local out = {}
    for k = 1, n do
        out[k] = pos_buf[k - 1]
    end
    return out
end

--- The scored alignment of `query` in `text` as 1-based BYTE columns, ascending (every byte of each matched
--- character) — what a completion menu highlights. Nil when `text` does not match.
---@param text string
---@param query string
---@param opts LvimFuzzyOpts?
---@return integer[]?
function M.positions(text, query, opts)
    return positions_of("lvimfuzzy_positions", text, query, opts)
end

--- What a list highlights in a matching `text`: the alignment plus every other occurrence of each term, as
--- 0-based CHARACTER indices, ascending. Nil when `text` does not match.
---@param text string
---@param query string
---@param opts LvimFuzzyOpts?
---@return integer[]?
function M.highlights(text, query, opts)
    return positions_of("lvimfuzzy_highlights", text, query, opts)
end

-- ─── blob contexts: a process's output straight in ───────────────────────────
-- For a STREAMING finder over a huge tree: the producer's raw stdout bytes go straight to a native context
-- that splits them on '\n' (empty lines skipped); no Lua string per candidate. The consumer materialises
-- only the rows it shows (`blob_text`).
---@class LvimFuzzyBlob
---@field id integer
---@field n integer
---@field gen integer     bumped by every blob_match; a superseded scheduled match is dropped
---@field _gc ffi.cdata*?

--- A new blob context (nil when the library is not loaded).
---@return LvimFuzzyBlob?
function M.blob_new()
    if not load_native() then
        return nil
    end
    local id, guard = new_handle(true)
    return { id = id, n = 0, gen = 0, _gc = guard }
end

--- Ingest a raw byte chunk (any boundaries). Returns the candidate count.
---@param blob LvimFuzzyBlob
---@param data string
---@return integer
function M.blob_append(blob, data)
    if blob._gc and lib then
        blob.n = lib.lvimfuzzy_ctx_append_blob(blob.id, data, #data)
    end
    return blob.n
end

--- The end of the stream (a last line without '\n'). Returns the candidate count.
---@param blob LvimFuzzyBlob
---@return integer
function M.blob_flush(blob)
    if blob._gc and lib then
        blob.n = lib.lvimfuzzy_ctx_flush(blob.id)
    end
    return blob.n
end

---@param blob LvimFuzzyBlob
---@return integer
function M.blob_count(blob)
    if blob._gc and lib then
        return lib.lvimfuzzy_ctx_count(blob.id)
    end
    return 0
end

--- The text of candidate `idx` (1-based), or nil.
---@param blob LvimFuzzyBlob
---@param idx integer
---@return string?
function M.blob_text(blob, idx)
    if not (blob._gc and lib) then
        return nil
    end
    local p = lib.lvimfuzzy_ctx_text(blob.id, idx, len_buf)
    if p == nil then
        return nil
    end
    return ffi.string(p, len_buf[0])
end

--- Rank `blob` on a scheduled tick: `on_result(results, count, total)` — `total` is every candidate that
--- matched (the whole pool for an empty query). A newer blob_match on the same blob supersedes this one.
--- `opts.scope` = what a grep result is searched by ("line", "path", "file"; default the whole row).
---@param blob LvimFuzzyBlob
---@param query string
---@param on_result fun(results: { index: integer, score: integer }[], count: integer, total: integer)
---@param opts LvimFuzzyOpts?
function M.blob_match(blob, query, on_result, opts)
    blob.gen = (blob.gen or 0) + 1
    local mygen = blob.gen
    vim.schedule(function()
        if mygen ~= blob.gen then
            return
        end
        if not (blob._gc and lib) then
            on_result({}, 0, 0)
            return
        end
        on_result(run(blob.id, M.blob_count(blob), query, opts))
    end)
end

--- Drop a blob context (native memory reclaimed). Safe to call more than once.
---@param blob LvimFuzzyBlob?
function M.blob_free(blob)
    if blob and blob._gc then
        ffi.gc(blob._gc, nil)
        blob._gc = nil
        if lib then
            lib.lvimfuzzy_ctx_drop(blob.id)
        end
        blob.n = 0
        blob.gen = (blob.gen or 0) + 1
    end
end

-- ─── introspection ───────────────────────────────────────────────────────────

--- "native-avx2", "native-scalar", or "none" (no usable library).
---@return string
function M.backend()
    if load_native() then
        return (lib.lvimfuzzy_avx2() == 1) and "native-avx2" or "native-scalar"
    end
    return "none"
end

--- True when the native library is loaded and its ABI matches.
---@return boolean
function M.native_loaded()
    return load_native() ~= nil
end

--- Why the library is not loaded (nil when it is).
---@return string?
function M.load_error()
    load_native()
    return load_error
end

--- The path the library was loaded from, or nil.
---@return string?
function M.library_path()
    load_native()
    return lib_path
end

--- The ABI this Lua needs.
---@return integer
function M.abi()
    return ABI
end

--- The loaded library's ABI (nil when none is loaded).
---@return integer?
function M.native_abi()
    if load_native() then
        return lib.lvimfuzzy_abi()
    end
    return nil
end

-- Feature probes of the ABI 1–6 era. Every feature is part of ABI 7, so they only say whether the library
-- is loaded; kept so callers written against them keep working.
M.has_append = M.native_loaded
M.has_chunk = M.native_loaded
M.has_par = M.native_loaded
M.has_blob = M.native_loaded
M.has_boost = M.native_loaded

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
    -- a changed library_path is probed again
    if opts and opts.library_path then
        probed, lib, lib_path, load_error = false, nil, nil, nil
    end
end

return M
