-- lvim-fuzzy: :checkhealth lvim-fuzzy.
-- Reports which backend is active (native AVX2 / native scalar / pure-Lua), where
-- the .so was found and whether its ABI matches, whether AVX2 is available, and
-- validates the config. When BOTH backends are available it runs a live PARITY
-- check — ranks a sample set through the native and the Lua matcher and diffs the
-- results — so a divergence between them can never go unnoticed. Read-only.
--
---@module "lvim-fuzzy.health"

local config = require("lvim-fuzzy.config")
local fuzzy = require("lvim-fuzzy")
local fallback = require("lvim-fuzzy.fallback")

local M = {}

--- Validate the config values, reporting each problem.
---@param h table  the vim.health reporter
local function check_config(h)
    local ok = true
    if config.backend ~= "auto" and config.backend ~= "native" and config.backend ~= "lua" then
        h.error(("config.backend must be 'auto'|'native'|'lua' (got %s)"):format(vim.inspect(config.backend)))
        ok = false
    end
    if type(config.max_results) ~= "number" or config.max_results < 0 then
        h.error("config.max_results must be a number ≥ 0")
        ok = false
    end
    if config.library_path ~= nil and type(config.library_path) ~= "string" then
        h.error("config.library_path must be a string path or nil")
        ok = false
    end
    if ok then
        h.ok("config is valid")
    end
end

--- Build a small deterministic identifier-like sample set for the parity check.
---@return string[]
local function sample_candidates()
    local stems = {
        "get",
        "set",
        "update",
        "buffer",
        "window",
        "config",
        "handler",
        "source",
        "complete",
        "cursor",
        "match",
        "score",
    }
    local out = {}
    local seed = 1
    local function rnd(m)
        seed = (seed * 1103515245 + 12345) % 2147483648
        return (seed % m) + 1
    end
    for _ = 1, 800 do
        local a, b = stems[rnd(#stems)], stems[rnd(#stems)]
        local style = rnd(3)
        if style == 1 then
            out[#out + 1] = a .. b:sub(1, 1):upper() .. b:sub(2)
        elseif style == 2 then
            out[#out + 1] = a .. "_" .. b
        else
            out[#out + 1] = a .. "." .. b
        end
    end
    return out
end

--- Rank a candidate set through the native library directly (stateless), for the
--- parity diff. Returns {index,score} pairs (1-based indices).
---@param cands string[]
---@param query string
---@return { index: integer, score: integer }[]
local function native_rank(cands, query)
    -- fuzzy.filter uses the native stateless entry when the lib is loaded
    local res = fuzzy.filter(cands, query)
    return res
end

--- Live native-vs-Lua parity: rank the sample through both and diff.
---@param h table
local function check_parity(h)
    if not fuzzy.native_loaded() then
        h.info("native library not loaded — parity check skipped (Lua backend is the only path)")
        return
    end
    local cands = sample_candidates()
    local queries = { "co", "buf", "conf", "gtcl", "handler", "get", "s", "cs" }
    for _, q in ipairs(queries) do
        local nat = native_rank(cands, q)
        local lua = fallback.match(cands, q, config.max_results or 0)
        if #nat ~= #lua then
            h.error(("PARITY: query %q count %d (native) ≠ %d (lua)"):format(q, #nat, #lua))
            return
        end
        for i = 1, #nat do
            if nat[i].index ~= lua[i].index or nat[i].score ~= lua[i].score then
                h.error(
                    ("PARITY: query %q rank %d differs — native(%d,%d) vs lua(%d,%d)"):format(
                        q,
                        i,
                        nat[i].index,
                        nat[i].score,
                        lua[i].index,
                        lua[i].score
                    )
                )
                return
            end
        end
    end
    h.ok("native and Lua backends are byte-identical on the sample set (8 queries)")
end

--- Entry point for `:checkhealth lvim-fuzzy`.
function M.check()
    local h = vim.health
    h.start("lvim-fuzzy")

    check_config(h)

    local backend = fuzzy.backend()
    if backend == "native-avx2" then
        h.ok(("backend: native (AVX2 SIMD) — %s"):format(fuzzy.library_path() or "?"))
    elseif backend == "native-scalar" then
        h.ok(("backend: native (scalar; CPU has no AVX2) — %s"):format(fuzzy.library_path() or "?"))
        h.info("an AVX2-capable x86_64 CPU enables the faster inter-candidate SIMD path")
    else
        if config.backend == "native" then
            h.error("backend forced to 'native' but the library is not loadable — build it with `sh native/build.sh`")
        elseif config.backend == "lua" then
            h.ok("backend: pure-Lua (forced by config.backend = 'lua')")
        else
            h.warn("backend: pure-Lua fallback — native library not found")
            h.info("build the native matcher for full speed: `sh native/build.sh` (needs a Rust toolchain)")
        end
    end

    -- ABI report — the loader accepts any library reporting ABI ≥ abi_min()
    if fuzzy.native_loaded() then
        local nabi = fuzzy.native_abi()
        if nabi and nabi >= fuzzy.abi_min() then
            h.ok(("ABI %d (accepted; this Lua understands up to %d)"):format(nabi, fuzzy.abi()))
            if fuzzy.has_append() then
                h.ok("incremental append supported (ABI ≥ 2)")
            else
                h.info("library predates incremental append (ABI < 2) — append falls back to a full re-prepare")
            end
            if fuzzy.has_chunk() then
                h.ok("cooperative async match supported (ABI ≥ 3) — a huge-pool match never blocks the UI")
            else
                h.info("library predates the chunked async match (ABI < 3) — match_async runs synchronously")
            end
            if fuzzy.has_par() then
                h.ok(
                    "parallel match supported (ABI ≥ 4) — the scan fans out across CPU cores (millions of "
                        .. "candidates in tens of ms)"
                )
            else
                h.info("library predates the parallel match (ABI < 4) — match runs single-threaded / chunked")
            end
            if fuzzy.has_blob() then
                h.ok(
                    "blob ingestion supported (ABI ≥ 5) — a streaming consumer feeds raw bytes straight to the "
                        .. "matcher (no per-candidate Lua strings; a multi-million-line pool loads without GC pauses)"
                )
            else
                h.info("library predates blob ingestion (ABI < 5) — streaming consumers use the per-string path")
            end
            if fuzzy.has_boost() then
                h.ok(
                    "boosted match supported (ABI ≥ 6) — a per-candidate score-offset array folds exact-prefix / "
                        .. "proximity signals into the rank before the cap (lvim-cmp)"
                )
            else
                h.info("library predates boosted match (ABI < 6) — callers' boosts are ignored (plain fuzzy rank)")
            end
        else
            h.error(("ABI too old: library reports %s, need ≥ %d"):format(tostring(nabi), fuzzy.abi_min()))
        end
    end

    check_parity(h)
end

return M
