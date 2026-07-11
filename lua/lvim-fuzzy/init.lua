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
--   • require("lvim-fuzzy").backend()  → "native-avx2" | "native-scalar" | "lua"
--
---@module "lvim-fuzzy"

local ffi = require("ffi")
local config = require("lvim-fuzzy.config")
local fallback = require("lvim-fuzzy.fallback")

local M = {}

-- ABI version this Lua was written against. The loader refuses a .so reporting a
-- different value (a stale/incompatible build) and uses the Lua fallback instead.
local ABI = 1

ffi.cdef([[
int lvimfuzzy_abi(void);
int lvimfuzzy_avx2(void);
int lvimfuzzy_prepare(const char *hay, const int *offs, int n);
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
                if abi_ok and abi == ABI then
                    lib = loaded
                    lib_path = p
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

-- ─── public API ──────────────────────────────────────────────────────────────

--- Build a matching context for a fixed candidate set. Do this ONCE per
--- completion / pick; then call `match(query, ctx)` on every keystroke.
---@param candidates string[]  the haystack (searched by their bytes)
---@return LvimFuzzyContext
function M.prepare(candidates)
    local n = #candidates
    ---@class LvimFuzzyContext
    ---@field candidates string[]
    ---@field n integer
    ---@field native boolean
    ---@field token integer
    ---@field blob? string
    ---@field offs? ffi.cdata*
    local ctx = { candidates = candidates, n = n, native = false, token = 0 }

    if load_native() and lib then
        -- concatenated bytes + (n+1) offsets; native prepare COPIES the blob
        local blob = table.concat(candidates)
        local offs = ffi.new("int[?]", n + 1)
        local pos = 0
        for i = 1, n do
            offs[i - 1] = pos
            pos = pos + #candidates[i]
        end
        offs[n] = pos
        token_seq = token_seq + 1
        ctx.native = true
        ctx.token = token_seq
        ctx.blob = blob
        ctx.offs = offs
        lib.lvimfuzzy_prepare(blob, offs, n)
        active_token = ctx.token
    else
        maybe_warn_fallback()
    end
    return ctx
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
    -- re-upload this set if another ctx became active since it was prepared
    if active_token ~= ctx.token and ctx.blob and ctx.offs then
        lib.lvimfuzzy_prepare(ctx.blob, ctx.offs, ctx.n)
        active_token = ctx.token
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

--- The ABI version this Lua expects.
---@return integer
function M.abi()
    return ABI
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
    active_token = nil
end

return M
